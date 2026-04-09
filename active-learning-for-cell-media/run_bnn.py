#!/usr/bin/env python3
"""
run_bnn.py — Full BNN benchmark runner (BNN_Regular + BNN_Hard).

Runs the complete pipeline:
  Phase 1: n_comp=10, n_synth=10 (fast initial pass)
  Phase 2: top-up to n_comp=30, n_synth=30

Features:
  - Per-job checkpointing → resume-safe if killed
  - nohup-friendly (all output to log files)
  - GPU-accelerated BNN surrogates via PyTorch MC Dropout
  - Pre-generates synthetic datasets once per (combo, dataset) pair

Usage:
    nohup python run_bnn.py --combo BNN_Regular --n-jobs 2 > /dev/null 2>&1 &
    nohup python run_bnn.py --combo BNN_Hard    --n-jobs 2 > /dev/null 2>&1 &
"""

import sys, os, time, json, argparse, logging, shutil, copy, traceback
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

# Suppress library INFO spam
logging.disable(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@contextmanager
def silence_stdout():
    old_out, old_err = sys.stdout, sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


# ── Lazy imports (done inside functions to keep top-level fast) ──────────
def _lazy_imports():
    from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import (
        run_harder_competition, HarderHideLabelCompetition
    )
    from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition
    from utils.datasets import load_dataset
    return run_harder_competition, HarderHideLabelCompetition, run_harder_open_race_competition, load_dataset


# ── Configuration ────────────────────────────────────────────────────────
DATASETS = [
    'MOBO_dataset_rat_myocyte',
    'DBO_dataset_rat_myocyte',
    'df_Human_Hela_regular_mode',
    'df_Human_Hela_timesaving_mode',
    'df_Human_T_Cell_Expanded',
    'df_Human_TF_Cell_Expanded',
]

OPTIMISERS = [
    'RANDOM',
    'BO_GP_EI', 'SBO_GP_PV', 'SBO_GP_EI_TRUNCDE',
    'SBO_ANN_PV', 'SMART_BO',
    'SBO_POLY_PV',
    'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'LATIN_HYPERCUBE',
    'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'PLACKETT_BURMAN',
    'DE_DIRECT',
]

NO_DIFF = {'noise_level': 0.0, 'heteroscedastic': False, 'n_modes': 1, 'mode_separation': 2.0}
HARD    = {'noise_level': 0.1, 'heteroscedastic': True,  'n_modes': 2, 'mode_separation': 2.0}

COMBO_MAP = {
    'BNN_Regular': ('bayesian_neural_network', NO_DIFF),
    'BNN_Hard':    ('bayesian_neural_network', HARD),
}

HIDDEN_FRACTIONS = [0.95, 0.99]
BATCH_SIZES      = [1, 10, 20]

PHASE1_N_COMP  = 10
PHASE1_N_SYNTH = 10
FINAL_N_COMP   = 30
FINAL_N_SYNTH  = 30

HTL_PLOT_DIR = ROOT / 'Plotting' / 'Hard_Mode' / 'Hide_The_Label'
OR_PLOT_DIR  = ROOT / 'Plotting' / 'Hard_Mode' / 'Open_Race'


def job_key(combo, dataset, hf, protocol, batch):
    return f"{combo}|{dataset}|hf{hf}|{protocol}|B{batch}"


def copy_new_plots(plot_dir, dest_dir, before_time):
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if plot_dir.exists():
        for f in sorted(plot_dir.glob('*.png')):
            if f.stat().st_mtime > before_time:
                shutil.copy2(f, dest_dir / f.name)
                count += 1
    return count


def pregenerate_synthetic_datasets(dataset_name, surrogate_type, difficulty_config,
                                   pool_size, n_synthetic, random_state=42):
    _, HarderHideLabelCompetition, _, load_dataset = _lazy_imports()

    competition = HarderHideLabelCompetition(
        surrogate_type=surrogate_type,
        difficulty_config=difficulty_config,
        random_state=random_state,
        use_gpu=True,
    )
    X, y, y_var, dataset = load_dataset(dataset_name)
    surrogate_model, scaler = competition.fit_surrogate_model(
        X, y, y_var, dataset_name, use_cache_model=True
    )
    datasets = []
    for synth_idx in range(n_synthetic):
        synthetic_seed = random_state + synth_idx * 1000
        X_pool, y_pool = competition.generate_candidate_pool(
            surrogate_model, scaler, X, n_points=pool_size, synthetic_seed=synthetic_seed
        )
        datasets.append((X_pool, y_pool))
    return datasets


# ── Result file finders ──────────────────────────────────────────────────
def find_htl_json(combo, dataset, hf, batch):
    hf_tag = f"hiddenfrac{int(hf * 100)}"
    complete_dir = ROOT / 'New_Results_Official' / combo / hf_tag / 'Hide_The_Label' / 'complete'
    if not complete_dir.exists():
        return None
    for f in complete_dir.glob('*.json'):
        try:
            d = json.load(open(f))
            tr = d.get('tournament_results', [])
            if not isinstance(tr, list) or not tr:
                continue
            t0 = tr[0]
            if dataset not in t0.get('dataset_name', ''):
                continue
            if abs(t0.get('hidden_fraction', 0) - hf) > 0.01:
                continue
            comps = t0.get('competitions', [])
            if not comps:
                continue
            first_opt = list(comps[0].get('optimizer_results', {}).values())
            if first_opt:
                hist = first_opt[0].get('optimization_history', [])
                if hist:
                    step0 = [h for h in hist if h.get('step') == 0]
                    inferred_batch = len(step0) if step0 else 1
                    if inferred_batch != batch:
                        continue
            return f
        except Exception:
            continue
    return None


def find_or_json(combo, dataset, batch):
    complete_dir = ROOT / 'New_Results_Official' / combo / 'Open_Race' / 'complete'
    if not complete_dir.exists():
        return None
    for f in complete_dir.glob('*.json'):
        if dataset not in f.name:
            continue
        try:
            d = json.load(open(f))
            tr = d.get('tournament_results', {})
            if not isinstance(tr, dict):
                continue
            b = tr.get('B')
            if b is None:
                b = tr.get('competition_settings', {}).get('B')
            if b is not None and b != batch:
                continue
            return f
        except Exception:
            continue
    return None


# ── Recompute analysis helpers ───────────────────────────────────────────
def recompute_htl_analysis(tournament_results):
    from collections import defaultdict
    all_opt = set()
    opt_data = defaultdict(lambda: {'steps': [], 'success': [], 'final_best': []})
    for tr in tournament_results:
        for comp in tr.get('competitions', []):
            for name, res in comp.get('optimizer_results', {}).items():
                all_opt.add(name)
                opt_data[name]['steps'].append(res.get('steps_taken', float('inf')))
                opt_data[name]['success'].append(1.0 if res.get('found_target', False) else 0.0)
                opt_data[name]['final_best'].append(res.get('final_best_value', 0.0))
    stats = {}
    for name in sorted(all_opt):
        d = opt_data[name]
        s, sc, fb = np.array(d['steps']), np.array(d['success']), np.array(d['final_best'])
        stats[name] = {
            'mean_steps': float(np.mean(s)), 'std_steps': float(np.std(s)),
            'median_steps': float(np.median(s)), 'success_rate': float(np.mean(sc)),
            'mean_final_best': float(np.mean(fb)), 'n_competitions': len(s),
        }
    ranked = sorted(stats, key=lambda n: stats[n]['mean_steps'])
    total = sum(len(tr.get('competitions', [])) for tr in tournament_results)
    return {
        'optimizer_names': sorted(all_opt), 'optimizer_stats': stats, 'ranking': ranked,
        'n_competitions': total, 'n_synthetic_datasets': len(tournament_results),
        'dataset_name': tournament_results[0].get('dataset_name') if tournament_results else None,
        'hidden_fraction': tournament_results[0].get('hidden_fraction') if tournament_results else None,
    }


def recompute_or_analysis(tournament_results):
    from collections import defaultdict
    comps = tournament_results.get('competitions', []) if isinstance(tournament_results, dict) else []
    opt_data = defaultdict(lambda: {'steps': [], 'regret': []})
    for comp in comps:
        for name, res in comp.get('optimizer_results', {}).items():
            opt_data[name]['steps'].append(res.get('steps', res.get('n_steps', 0)))
            opt_data[name]['regret'].append(res.get('final_regret', res.get('regret', 0.0)))
    stats = {}
    for name in sorted(opt_data):
        d = opt_data[name]
        stats[name] = {
            'mean_steps': float(np.mean(d['steps'])), 'std_steps': float(np.std(d['steps'])),
            'mean_regret': float(np.mean(d['regret'])), 'n_competitions': len(d['steps']),
        }
    return {'optimizer_names': sorted(opt_data), 'optimizer_stats': stats, 'n_competitions': len(comps)}


# ── Bridge: copy Phase 1 JSONs from Results/ → complete/ ────────────────
def _bridge_phase1_to_complete(combo, surrogate, diff, status, log):
    """Copy Phase 1 output JSONs from Results/Hard_Mode/ into the
    New_Results_Official/<combo>/.../complete/ layout that Phase 2 expects.

    Phase 1 calls the experiment functions with save_results=True, which
    writes to Results/Hard_Mode/{Hide_The_Label,Open_Race}/.  Phase 2's
    find_htl_json / find_or_json read from
    New_Results_Official/<combo>/<hf>/Hide_The_Label/complete/ (and
    .../Open_Race/complete/).  This function bridges the two layouts.
    """
    htl_src = ROOT / 'Results' / 'Hard_Mode' / 'Hide_The_Label'
    or_src  = ROOT / 'Results' / 'Hard_Mode' / 'Open_Race'
    n_copied = 0

    for dataset in DATASETS:
        for hf in HIDDEN_FRACTIONS:
            hf_tag = f"hiddenfrac{int(hf*100)}"
            for batch in BATCH_SIZES:
                p1_k = f"p1|{job_key(combo, dataset, hf, 'HTL', batch)}"
                br_k = f"bridge|{job_key(combo, dataset, hf, 'HTL', batch)}"
                if status.get(p1_k) == 'done' and status.get(br_k) != 'done':
                    # Find matching HTL JSON in Results/
                    if htl_src.exists():
                        for f in sorted(htl_src.glob(f"{dataset}*hide_the_label*.json"),
                                        key=lambda p: p.stat().st_mtime, reverse=True):
                            try:
                                d = json.load(open(f))
                                if d.get('surrogate_type') != surrogate:
                                    continue
                                tr = d.get('tournament_results', [])
                                if not isinstance(tr, list) or not tr:
                                    continue
                                if abs(tr[0].get('hidden_fraction', 0) - hf) > 0.01:
                                    continue
                                dest_dir = (ROOT / 'New_Results_Official' / combo
                                            / hf_tag / 'Hide_The_Label' / 'complete')
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(f, dest_dir / f.name)
                                status[br_k] = 'done'
                                n_copied += 1
                                log(f"BRIDGE HTL hf={hf} B={batch} [{dataset}] → {dest_dir.name}/{f.name}")
                                break
                            except Exception:
                                continue

                # OR (no hf nesting in dest)
                p1_k = f"p1|{job_key(combo, dataset, hf, 'OR', batch)}"
                br_k = f"bridge|{job_key(combo, dataset, hf, 'OR', batch)}"
                if status.get(p1_k) == 'done' and status.get(br_k) != 'done':
                    if or_src.exists():
                        for f in sorted(or_src.glob(f"{dataset}*open_race*.json"),
                                        key=lambda p: p.stat().st_mtime, reverse=True):
                            try:
                                d = json.load(open(f))
                                if d.get('surrogate_type') != surrogate:
                                    continue
                                tr = d.get('tournament_results', {})
                                if isinstance(tr, dict) and tr.get('B') == batch:
                                    dest_dir = (ROOT / 'New_Results_Official' / combo
                                                / 'Open_Race' / 'complete')
                                    dest_dir.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(f, dest_dir / f.name)
                                    status[br_k] = 'done'
                                    n_copied += 1
                                    log(f"BRIDGE OR  B={batch} [{dataset}] → {dest_dir.name}/{f.name}")
                                    break
                            except Exception:
                                continue

    log(f"BRIDGE COMPLETE. {n_copied} files copied to complete/.")


# ── Phase 1: Initial run (n=10) ─────────────────────────────────────────
def run_phase1(combo, surrogate, diff, n_jobs, status, save_status, log):
    run_harder_competition, _, run_harder_open_race_competition, _ = _lazy_imports()

    out_base = ROOT / 'New_Results_Official' / combo
    total = len(DATASETS) * len(HIDDEN_FRACTIONS) * len(BATCH_SIZES) * 2
    done = sum(1 for k, v in status.items() if k.startswith('p1|') and v == 'done')
    log(f"PHASE 1 — initial run (n_comp={PHASE1_N_COMP}, n_synth={PHASE1_N_SYNTH}). {done}/{total} done.")

    for dataset in DATASETS:
        # Check if any HTL job needs running
        htl_needs = any(
            status.get(f"p1|{job_key(combo, dataset, hf, 'HTL', b)}") != 'done'
            for hf in HIDDEN_FRACTIONS for b in BATCH_SIZES
        )
        synth_data = None
        if htl_needs:
            log(f"Pre-generating {PHASE1_N_SYNTH} synthetics for {dataset}...")
            t0 = time.time()
            try:
                with silence_stdout():
                    synth_data = pregenerate_synthetic_datasets(
                        dataset, surrogate, diff,
                        pool_size=200, n_synthetic=PHASE1_N_SYNTH, random_state=42,
                    )
                log(f"Synthetics ready in {time.time()-t0:.1f}s")
            except Exception as e:
                log(f"FAILED synthetics: {e}")
                traceback.print_exc()
                synth_data = None

        for hf in HIDDEN_FRACTIONS:
            hf_tag = f"hiddenfrac{int(hf*100)}"
            for batch in BATCH_SIZES:
                # ── HTL ──
                k = f"p1|{job_key(combo, dataset, hf, 'HTL', batch)}"
                if status.get(k) != 'done':
                    log(f"HTL hf={hf} B={batch} [{dataset}]")
                    htl_dest = out_base / hf_tag / 'Hide_The_Label'
                    t_before = time.time()
                    try:
                        with silence_stdout():
                            run_harder_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=OPTIMISERS,
                                n_competitions=PHASE1_N_COMP,
                                hidden_fraction=hf,
                                pool_size=200,
                                batch_size=batch,
                                n_jobs=n_jobs,
                                random_state=42,
                                save_results=True,
                                num_of_synthetic_data=PHASE1_N_SYNTH,
                                use_gpu=True,
                                pregenerated_synthetic_datasets=synth_data,
                            )
                        copy_new_plots(HTL_PLOT_DIR, htl_dest, t_before)
                        done += 1
                        log(f"HTL DONE in {time.time()-t_before:.0f}s ({done}/{total})")
                        status[k] = 'done'
                    except Exception as e:
                        status[k] = f'failed: {e}'
                        log(f"HTL FAILED: {e}")
                        traceback.print_exc()
                    save_status()

                # ── OR ──
                k = f"p1|{job_key(combo, dataset, hf, 'OR', batch)}"
                if status.get(k) != 'done':
                    log(f"OR  B={batch} [{dataset}]")
                    or_dest = out_base / hf_tag / 'Open_Race'
                    t_before = time.time()
                    try:
                        with silence_stdout():
                            run_harder_open_race_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=OPTIMISERS,
                                n_competitions=PHASE1_N_COMP,
                                S=200, R=10, B=batch,
                                n_jobs=n_jobs,
                                random_state=42,
                                save_results=True,
                            )
                        copy_new_plots(OR_PLOT_DIR, or_dest, t_before)
                        done += 1
                        log(f"OR  DONE in {time.time()-t_before:.0f}s ({done}/{total})")
                        status[k] = 'done'
                    except Exception as e:
                        status[k] = f'failed: {e}'
                        log(f"OR  FAILED: {e}")
                        traceback.print_exc()
                    save_status()

        synth_data = None  # free memory

    log(f"PHASE 1 COMPLETE. {sum(1 for k,v in status.items() if k.startswith('p1|') and v=='done')}/{total} done.")


# ── Phase 2: Top-up to n=30 ─────────────────────────────────────────────
def run_phase2(combo, surrogate, diff, n_jobs, status, save_status, log):
    run_harder_competition, _, run_harder_open_race_competition, _ = _lazy_imports()

    total_htl = len(DATASETS) * len(HIDDEN_FRACTIONS) * len(BATCH_SIZES)
    total_or  = len(DATASETS) * len(BATCH_SIZES)
    total = total_htl + total_or
    done = sum(1 for k, v in status.items() if k.startswith('p2|') and v == 'done')
    log(f"PHASE 2 — top-up to (n_comp={FINAL_N_COMP}, n_synth={FINAL_N_SYNTH}). {done}/{total} done.")

    for dataset in DATASETS:
        # Pre-generate all 30 synthetic datasets once
        log(f"Pre-generating {FINAL_N_SYNTH} synthetics for {dataset}...")
        t0 = time.time()
        all_synth = None
        try:
            with silence_stdout():
                all_synth = pregenerate_synthetic_datasets(
                    dataset, surrogate, diff,
                    pool_size=200, n_synthetic=FINAL_N_SYNTH, random_state=42,
                )
            log(f"Synthetics ready in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"FAILED synthetics: {e}")
            traceback.print_exc()

        or_done_for_batch = set()  # track OR dedup across hidden fractions

        for hf in HIDDEN_FRACTIONS:
            for batch in BATCH_SIZES:
                # ── HTL top-up ──
                k = f"p2|{job_key(combo, dataset, hf, 'HTL', batch)}"
                if status.get(k) != 'done':
                    log(f"HTL TOPUP hf={hf} B={batch} [{dataset}]")
                    t_before = time.time()
                    try:
                        existing_path = find_htl_json(combo, dataset, hf, batch)
                        if existing_path is None:
                            raise FileNotFoundError(f"No HTL JSON for {dataset} hf={hf} B={batch}")

                        with open(existing_path) as f:
                            existing = json.load(f)

                        ext_tr = existing.get('tournament_results', [])
                        cur_ns = len(ext_tr)
                        cur_nc = len(ext_tr[0].get('competitions', [])) if ext_tr else 0

                        if cur_ns >= FINAL_N_SYNTH and cur_nc >= FINAL_N_COMP:
                            log(f"HTL already at target ({cur_ns}s×{cur_nc}c)")
                            status[k] = 'done'
                            save_status()
                            done += 1
                            continue

                        # Step A: more competitions on existing synthetic datasets
                        if cur_nc < FINAL_N_COMP and all_synth:
                            existing_synths = all_synth[:cur_ns]
                            with silence_stdout():
                                sup, _ = run_harder_competition(
                                    dataset_name=dataset,
                                    surrogate_type=surrogate,
                                    difficulty_config=diff,
                                    optimizer_names=OPTIMISERS,
                                    n_competitions=FINAL_N_COMP,
                                    hidden_fraction=hf,
                                    pool_size=200, batch_size=batch,
                                    n_jobs=n_jobs, random_state=42,
                                    save_results=False,
                                    num_of_synthetic_data=cur_ns,
                                    use_gpu=True,
                                    pregenerated_synthetic_datasets=existing_synths,
                                )
                            sup_tr = sup.get('tournament_results', [])
                            for i in range(cur_ns):
                                new_comps = sup_tr[i].get('competitions', [])
                                ext_tr[i]['competitions'].extend(new_comps[cur_nc:])
                                ext_tr[i]['n_competitions'] = len(ext_tr[i]['competitions'])

                        # Step B: new synthetic datasets with full competitions
                        if cur_ns < FINAL_N_SYNTH and all_synth:
                            new_synths = all_synth[cur_ns:]
                            with silence_stdout():
                                new_res, _ = run_harder_competition(
                                    dataset_name=dataset,
                                    surrogate_type=surrogate,
                                    difficulty_config=diff,
                                    optimizer_names=OPTIMISERS,
                                    n_competitions=FINAL_N_COMP,
                                    hidden_fraction=hf,
                                    pool_size=200, batch_size=batch,
                                    n_jobs=n_jobs, random_state=42,
                                    save_results=False,
                                    num_of_synthetic_data=len(new_synths),
                                    use_gpu=True,
                                    pregenerated_synthetic_datasets=new_synths,
                                )
                            ext_tr.extend(new_res.get('tournament_results', []))

                        existing['tournament_results'] = ext_tr
                        existing['num_synthetic_datasets'] = len(ext_tr)
                        existing['combined_analysis'] = recompute_htl_analysis(ext_tr)
                        existing['topup_applied'] = True

                        with open(existing_path, 'w') as f:
                            json.dump(existing, f, cls=NumpySafeEncoder)

                        final_ns = len(ext_tr)
                        final_nc = len(ext_tr[0].get('competitions', [])) if ext_tr else 0
                        done += 1
                        log(f"HTL TOPUP DONE in {time.time()-t_before:.0f}s ({done}/{total}) → {final_ns}s×{final_nc}c")
                        status[k] = 'done'
                    except Exception as e:
                        status[k] = f'failed: {e}'
                        log(f"HTL TOPUP FAILED: {e}")
                        traceback.print_exc()
                    save_status()

                # ── OR top-up (deduplicated across hidden fractions) ──
                or_dedup_key = f"{dataset}|B{batch}"
                k = f"p2|{job_key(combo, dataset, hf, 'OR', batch)}"
                if or_dedup_key not in or_done_for_batch and status.get(k) != 'done':
                    log(f"OR  TOPUP B={batch} [{dataset}]")
                    t_before = time.time()
                    try:
                        existing_path = find_or_json(combo, dataset, batch)
                        if existing_path is None:
                            raise FileNotFoundError(f"No OR JSON for {dataset} B={batch}")

                        with open(existing_path) as f:
                            existing = json.load(f)

                        ext_tr = existing.get('tournament_results', {})
                        ext_comps = ext_tr.get('competitions', []) if isinstance(ext_tr, dict) else []
                        cur_nc = len(ext_comps)

                        if cur_nc >= FINAL_N_COMP:
                            log(f"OR already at target ({cur_nc}c)")
                            status[k] = 'done'
                            or_done_for_batch.add(or_dedup_key)
                            save_status()
                            done += 1
                            continue

                        with silence_stdout():
                            sup_result = run_harder_open_race_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=OPTIMISERS,
                                n_competitions=FINAL_N_COMP,
                                S=200, R=10, B=batch,
                                n_jobs=n_jobs, random_state=42,
                                save_results=False,
                            )

                        if isinstance(sup_result, tuple):
                            sup_tr = sup_result[0]
                        else:
                            sup_tr = sup_result

                        new_comps = sup_tr.get('competitions', []) if isinstance(sup_tr, dict) else []
                        ext_comps.extend(new_comps[cur_nc:])

                        if isinstance(ext_tr, dict):
                            ext_tr['competitions'] = ext_comps
                            ext_tr['n_competitions'] = len(ext_comps)
                            if new_comps:
                                ext_tr['optimizer_names'] = list(
                                    new_comps[0].get('optimizer_results', {}).keys())
                        existing['tournament_results'] = ext_tr
                        existing['analysis'] = recompute_or_analysis(ext_tr)
                        existing['topup_applied'] = True

                        with open(existing_path, 'w') as f:
                            json.dump(existing, f, cls=NumpySafeEncoder)

                        done += 1
                        log(f"OR  TOPUP DONE in {time.time()-t_before:.0f}s ({done}/{total}) → {len(ext_comps)}c")
                        status[k] = 'done'
                        or_done_for_batch.add(or_dedup_key)
                    except Exception as e:
                        status[k] = f'failed: {e}'
                        log(f"OR  TOPUP FAILED: {e}")
                        traceback.print_exc()
                    save_status()
                elif or_dedup_key in or_done_for_batch:
                    status[k] = 'done'
                    save_status()

        all_synth = None  # free memory

    log(f"PHASE 2 COMPLETE. {sum(1 for k,v in status.items() if k.startswith('p2|') and v=='done')}/{total} done.")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', required=True, choices=list(COMBO_MAP.keys()))
    parser.add_argument('--n-jobs', type=int, default=2)
    args = parser.parse_args()

    combo = args.combo
    n_jobs = args.n_jobs
    surrogate, diff = COMBO_MAP[combo]

    # Status file for checkpoint/resume
    status_file = ROOT / 'New_Results_Official' / f'_bnn_status_{combo}.json'
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status = {}
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)

    def save_status():
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    # Log file
    log_file = ROOT / 'New_Results_Official' / f'_bnn_log_{combo}.txt'
    def log(msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] [{combo}] {msg}"
        print(line, flush=True)
        with open(log_file, 'a') as f:
            f.write(line + '\n')

    log(f"=" * 60)
    log(f"BNN BENCHMARK START — combo={combo}, n_jobs={n_jobs}")
    log(f"Surrogate: {surrogate}, Difficulty: {diff}")
    log(f"Target: n_comp={FINAL_N_COMP}, n_synth={FINAL_N_SYNTH}")
    log(f"=" * 60)

    t_global = time.time()

    # Phase 1
    run_phase1(combo, surrogate, diff, n_jobs, status, save_status, log)

    # Bridge: copy Phase 1 output into complete/ for Phase 2
    _bridge_phase1_to_complete(combo, surrogate, diff, status, log)
    save_status()

    # Phase 2
    run_phase2(combo, surrogate, diff, n_jobs, status, save_status, log)

    elapsed = time.time() - t_global
    p1_done = sum(1 for k, v in status.items() if k.startswith('p1|') and v == 'done')
    p1_fail = sum(1 for k, v in status.items() if k.startswith('p1|') and 'failed' in str(v))
    p2_done = sum(1 for k, v in status.items() if k.startswith('p2|') and v == 'done')
    p2_fail = sum(1 for k, v in status.items() if k.startswith('p2|') and 'failed' in str(v))
    log(f"ALL DONE in {elapsed/3600:.1f}h. Phase1: {p1_done} done/{p1_fail} fail. Phase2: {p2_done} done/{p2_fail} fail.")


if __name__ == '__main__':
    main()
