#!/usr/bin/env python3
"""
Top-up script: bump existing results from (n_synth=10, n_comp=10) to (n_synth=30, n_comp=30).

Strategy:
  HTL:
    - For existing 10 synthetic datasets: run 20 MORE competitions (indices 10-29)
      using the SAME pre-generated pool. The seed scheme guarantees indices 10-29
      produce different seeds than 0-9.
    - Generate 20 NEW synthetic datasets (indices 10-29, seeds 10042..29042)
      and run 30 competitions on each.
    - Merge into existing JSON → 30 synth × 30 comp.

  OR:
    - Run 20 more competitions (the seed scheme ensures new indices = new seeds)
    - Merge into existing JSON → 30 comp total.

Usage:
    python run_topup.py --combo GP_Regular --n-jobs 2
"""

import argparse
import json
import time
import copy
import sys
import os
import logging
import traceback
import multiprocessing
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed


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

# Suppress verbose logging from competition code — only show WARNING+
logging.disable(logging.INFO)

ROOT = Path(__file__).resolve().parent

# ── Settings ─────────────────────────────────────────────────────────────────
TARGET_N_SYNTH = 30
TARGET_N_COMP  = 30
CURRENT_N_SYNTH = 10
CURRENT_N_COMP  = 10

DATASETS = [
    'MOBO_dataset_rat_myocyte', 'DBO_dataset_rat_myocyte',
    'df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode',
    'df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded',
]
HIDDEN_FRACTIONS = [0.95, 0.99]
BATCH_SIZES      = [1, 10, 20]

NO_DIFF = {'noise_level': 0.0, 'heteroscedastic': False, 'n_modes': 1, 'mode_separation': 2.0}
HARD    = {'noise_level': 0.1, 'heteroscedastic': True,  'n_modes': 2, 'mode_separation': 2.0}

COMBO_MAP = {
    'GP_Regular': ('gaussian_process', NO_DIFF),
    'GP_Hard':    ('gaussian_process', HARD),
    'RF_Regular': ('random_forest',    NO_DIFF),
    'RF_Hard':    ('random_forest',    HARD),
    'NN_Regular':  ('neural_network',           NO_DIFF),
    'NN_Hard':     ('neural_network',           HARD),
    'BNN_Regular': ('bayesian_neural_network',  NO_DIFF),
    'BNN_Hard':    ('bayesian_neural_network',  HARD),
}

OPTIMISERS = [
    'RANDOM',
    'BO_GP_EI', 'SBO_GP_PV', 'SBO_GP_EI_TRUNCDE',
    'SBO_ANN_PV', 'SMART_BO',
    'SBO_POLY_PV',
    'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'LATIN_HYPERCUBE',
    'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'PLACKETT_BURMAN',
    'DE_DIRECT',
]


@contextmanager
def silence_stdout():
    old = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def job_key(combo, dataset, hf, protocol, batch):
    return f"{combo}|{dataset}|hf{hf}|{protocol}|B{batch}"


def pregenerate_synthetic_datasets(dataset_name, surrogate_type, difficulty_config,
                                   pool_size=200, n_synthetic=30, random_state=42,
                                   start_idx=0):
    """Generate synthetic datasets. start_idx allows generating only new ones."""
    from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import (
        HarderHideLabelCompetition, run_harder_competition
    )

    competition = HarderHideLabelCompetition(
        surrogate_type=surrogate_type,
        difficulty_config=difficulty_config,
        random_state=random_state, use_gpu=True
    )

    from utils.datasets import load_dataset
    X, y, y_var, dataset = load_dataset(dataset_name)
    surrogate_model, scaler = competition.fit_surrogate_model(X, y, y_var, dataset_name)

    datasets = []
    for synth_idx in range(start_idx, n_synthetic):
        synthetic_seed = random_state + synth_idx * 1000
        X_pool, y_pool = competition.generate_candidate_pool(
            surrogate_model, scaler, X, n_points=pool_size, synthetic_seed=synthetic_seed
        )
        datasets.append((X_pool, y_pool))

    return datasets


def find_existing_htl_json(combo, dataset, hf, batch):
    """Find the existing HTL JSON in New_Results_Official/.../complete/."""
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
            dn = t0.get('dataset_name', '')
            if dataset not in dn:
                continue
            # Check hidden_fraction
            if abs(t0.get('hidden_fraction', 0) - hf) > 0.01:
                continue
            # Check batch size from competitions
            comps = t0.get('competitions', [])
            if not comps:
                continue
            # Infer batch size from first competition
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


def find_existing_or_json(combo, dataset, batch):
    """Find the existing OR JSON in New_Results_Official/.../Open_Race/complete/."""
    complete_dir = ROOT / 'New_Results_Official' / combo / 'Open_Race' / 'complete'
    if not complete_dir.exists():
        return None
    # Map dataset names to filename patterns
    ds_to_pattern = {
        'MOBO_dataset_rat_myocyte': 'MOBO_dataset_rat_myocyte',
        'DBO_dataset_rat_myocyte': 'DBO_dataset_rat_myocyte',
        'df_Human_Hela_regular_mode': 'df_Human_Hela_regular_mode',
        'df_Human_Hela_timesaving_mode': 'df_Human_Hela_timesaving_mode',
        'df_Human_T_Cell_Expanded': 'df_Human_T_Cell_Expanded',
        'df_Human_TF_Cell_Expanded': 'df_Human_TF_Cell_Expanded',
    }
    pattern = ds_to_pattern.get(dataset, dataset)
    for f in complete_dir.glob('*.json'):
        fname = f.name
        if pattern not in fname:
            continue
        try:
            d = json.load(open(f))
            tr = d.get('tournament_results', {})
            if not isinstance(tr, dict):
                continue
            b = tr.get('B')
            if b is None:
                cs = tr.get('competition_settings', {})
                b = cs.get('B')
            if b is not None and b != batch:
                continue
            return f
        except Exception:
            continue
    return None


def run_htl_topup(combo, dataset, hf, batch, surrogate, diff, n_jobs, existing_path):
    """
    Top-up an existing HTL result from (10 synth, 10 comp) to (30 synth, 30 comp).

    Returns the merged result dict, or None on failure.
    """
    from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition

    with open(existing_path) as f:
        existing = json.load(f)

    ext_tr = existing.get('tournament_results', [])
    current_n_synth = len(ext_tr)
    current_n_comp = len(ext_tr[0].get('competitions', [])) if ext_tr else 0

    if current_n_synth >= TARGET_N_SYNTH and current_n_comp >= TARGET_N_COMP:
        return None  # already at target

    # ── Step 1: Generate ALL 30 synthetic datasets ────────────────────────
    all_synth = pregenerate_synthetic_datasets(
        dataset, surrogate, diff,
        pool_size=200, n_synthetic=TARGET_N_SYNTH, random_state=42, start_idx=0
    )

    # ── Step 2: For existing synth datasets (0..current_n_synth-1),
    #            run additional competitions (indices current_n_comp..TARGET_N_COMP-1) ──
    if current_n_comp < TARGET_N_COMP:
        extra_comps_needed = TARGET_N_COMP - current_n_comp
        existing_synths = all_synth[:current_n_synth]

        # Run with n_competitions = TARGET_N_COMP. The competition seed scheme is:
        #   seed = random_state + synth_hash + i*10000 + comp_hash
        # So competitions 0..9 will reproduce what we have, and 10..29 are new.
        # We run all TARGET_N_COMP and only keep indices [current_n_comp:]
        supplement_existing, _ = run_harder_competition(
            dataset_name=dataset,
            surrogate_type=surrogate,
            difficulty_config=diff,
            optimizer_names=OPTIMISERS,
            n_competitions=TARGET_N_COMP,
            hidden_fraction=hf,
            pool_size=200,
            batch_size=batch,
            n_jobs=n_jobs,
            random_state=42,
            save_results=False,
            num_of_synthetic_data=current_n_synth,
            use_gpu=True,
            pregenerated_synthetic_datasets=existing_synths,
        )

        sup_tr = supplement_existing.get('tournament_results', [])
        for i in range(current_n_synth):
            new_comps = sup_tr[i].get('competitions', [])
            # Append only the NEW competitions (indices current_n_comp onwards)
            ext_tr[i]['competitions'].extend(new_comps[current_n_comp:])
            ext_tr[i]['n_competitions'] = len(ext_tr[i]['competitions'])

    # ── Step 3: For NEW synth datasets (current_n_synth..TARGET_N_SYNTH-1),
    #            run all TARGET_N_COMP competitions ──
    if current_n_synth < TARGET_N_SYNTH:
        new_synths = all_synth[current_n_synth:]

        new_result, _ = run_harder_competition(
            dataset_name=dataset,
            surrogate_type=surrogate,
            difficulty_config=diff,
            optimizer_names=OPTIMISERS,
            n_competitions=TARGET_N_COMP,
            hidden_fraction=hf,
            pool_size=200,
            batch_size=batch,
            n_jobs=n_jobs,
            random_state=42,
            save_results=False,
            num_of_synthetic_data=len(new_synths),
            use_gpu=True,
            pregenerated_synthetic_datasets=new_synths,
        )

        new_tr = new_result.get('tournament_results', [])
        ext_tr.extend(new_tr)

    # ── Step 4: Recompute combined_analysis ──
    existing['tournament_results'] = ext_tr
    existing['num_synthetic_datasets'] = len(ext_tr)
    existing['combined_analysis'] = recompute_combined_analysis_htl(ext_tr)
    existing['topup_applied'] = True
    existing['topup_target'] = {'n_synth': TARGET_N_SYNTH, 'n_comp': TARGET_N_COMP}

    return existing


def run_or_topup(combo, dataset, batch, surrogate, diff, n_jobs, existing_path):
    """
    Top-up an existing OR result from 10 comp to 30 comp.
    """
    from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition

    with open(existing_path) as f:
        existing = json.load(f)

    ext_tr = existing.get('tournament_results', {})
    ext_comps = ext_tr.get('competitions', []) if isinstance(ext_tr, dict) else []
    current_n_comp = len(ext_comps)

    if current_n_comp >= TARGET_N_COMP:
        return None  # already at target

    # Run full TARGET_N_COMP competitions — seeds 0..9 reproduce existing, 10..29 are new
    sup_result = run_harder_open_race_competition(
        dataset_name=dataset,
        surrogate_type=surrogate,
        difficulty_config=diff,
        optimizer_names=OPTIMISERS,
        n_competitions=TARGET_N_COMP,
        S=200,
        R=10,
        B=batch,
        n_jobs=n_jobs,
        random_state=42,
        save_results=False,
    )

    # Unpack tuple if needed
    if isinstance(sup_result, tuple):
        sup_tr = sup_result[0]
    else:
        sup_tr = sup_result.get('tournament_results', {}) if isinstance(sup_result, dict) else sup_result

    if isinstance(sup_tr, dict):
        new_comps = sup_tr.get('competitions', [])
    else:
        new_comps = []

    # Append only the NEW competitions
    ext_comps.extend(new_comps[current_n_comp:])

    if isinstance(ext_tr, dict):
        ext_tr['competitions'] = ext_comps
        ext_tr['n_competitions'] = len(ext_comps)
        if new_comps:
            ext_tr['optimizer_names'] = list(new_comps[0].get('optimizer_results', {}).keys())

    existing['tournament_results'] = ext_tr
    # Recompute analysis
    existing['analysis'] = recompute_analysis_or(ext_tr)
    existing['topup_applied'] = True
    existing['topup_target'] = {'n_comp': TARGET_N_COMP}

    return existing


def recompute_combined_analysis_htl(tournament_results):
    """Recompute combined_analysis from all tournament results."""
    from collections import defaultdict
    import numpy as np

    all_opt_names = set()
    opt_data = defaultdict(lambda: {'steps': [], 'success': [], 'final_best': []})

    for tr in tournament_results:
        for comp in tr.get('competitions', []):
            for opt_name, opt_res in comp.get('optimizer_results', {}).items():
                all_opt_names.add(opt_name)
                opt_data[opt_name]['steps'].append(opt_res.get('steps_taken', float('inf')))
                opt_data[opt_name]['success'].append(1.0 if opt_res.get('found_target', False) else 0.0)
                opt_data[opt_name]['final_best'].append(opt_res.get('final_best_value', 0.0))

    all_opt_names = sorted(all_opt_names)
    optimizer_stats = {}
    for name in all_opt_names:
        d = opt_data[name]
        steps = np.array(d['steps'])
        success = np.array(d['success'])
        final_best = np.array(d['final_best'])
        optimizer_stats[name] = {
            'mean_steps': float(np.mean(steps)) if len(steps) > 0 else None,
            'std_steps': float(np.std(steps)) if len(steps) > 0 else None,
            'median_steps': float(np.median(steps)) if len(steps) > 0 else None,
            'success_rate': float(np.mean(success)) if len(success) > 0 else None,
            'mean_final_best': float(np.mean(final_best)) if len(final_best) > 0 else None,
            'n_competitions': len(steps),
        }

    ranked = sorted(
        optimizer_stats.keys(),
        key=lambda n: optimizer_stats[n]['mean_steps'] if optimizer_stats[n]['mean_steps'] is not None else 1e9
    )

    total_comps = sum(len(tr.get('competitions', [])) for tr in tournament_results)

    return {
        'optimizer_names': all_opt_names,
        'optimizer_stats': optimizer_stats,
        'ranking': ranked,
        'n_competitions': total_comps,
        'n_synthetic_datasets': len(tournament_results),
        'dataset_name': tournament_results[0].get('dataset_name') if tournament_results else None,
        'hidden_fraction': tournament_results[0].get('hidden_fraction') if tournament_results else None,
        'pool_size': tournament_results[0].get('pool_size') if tournament_results else None,
    }


def recompute_analysis_or(tournament_results):
    """Recompute analysis for OR results."""
    from collections import defaultdict
    import numpy as np

    comps = tournament_results.get('competitions', []) if isinstance(tournament_results, dict) else []

    all_opt_names = set()
    opt_data = defaultdict(lambda: {'steps': [], 'regret': []})

    for comp in comps:
        for opt_name, opt_res in comp.get('optimizer_results', {}).items():
            all_opt_names.add(opt_name)
            opt_data[opt_name]['steps'].append(opt_res.get('steps', opt_res.get('n_steps', 0)))
            opt_data[opt_name]['regret'].append(opt_res.get('final_regret', opt_res.get('regret', 0.0)))

    all_opt_names = sorted(all_opt_names)
    optimizer_stats = {}
    for name in all_opt_names:
        d = opt_data[name]
        steps = np.array(d['steps'])
        regret = np.array(d['regret'])
        optimizer_stats[name] = {
            'mean_steps': float(np.mean(steps)) if len(steps) > 0 else None,
            'std_steps': float(np.std(steps)) if len(steps) > 0 else None,
            'mean_regret': float(np.mean(regret)) if len(regret) > 0 else None,
            'n_competitions': len(steps),
        }

    return {
        'optimizer_names': all_opt_names,
        'optimizer_stats': optimizer_stats,
        'n_competitions': len(comps),
    }


def _htl_worker(args):
    """
    Top-level worker for ProcessPoolExecutor — runs ONE (hf, batch) HTL top-up.
    Returns (hf, batch, new_comps_by_synth, new_tr_entries, elapsed, err_str).
    Must be a top-level function so it can be pickled by multiprocessing.
    """
    import sys, os, logging, time, traceback
    from pathlib import Path
    from contextlib import contextmanager

    @contextmanager
    def _silence():
        old_out, old_err = sys.stdout, sys.stderr
        with open(os.devnull, 'w') as dn:
            sys.stdout = dn
            sys.stderr = dn
            try:
                yield
            finally:
                sys.stdout = old_out
                sys.stderr = old_err

    def _log(msg):
        """Write progress directly to fd 1 (log file), bypassing Python stdout redirection."""
        try:
            os.write(1, (msg + "\n").encode())
        except Exception:
            pass

    logging.disable(logging.CRITICAL)

    (dataset, hf, batch, surrogate, diff, all_synth_30,
     current_ns, current_nc, n_jobs_inner,
     OPTIMISERS_, TARGET_N_SYNTH_, TARGET_N_COMP_,
     root_str, optimizer_max_steps_) = args

    sys.path.insert(0, root_str)
    from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition

    tag = f"[Worker hf={hf} B={batch}]"
    t0 = time.time()
    try:
        # Apply optimizer step caps via environment variable (reliable in spawned processes)
        if optimizer_max_steps_:
            import json as _json
            os.environ['OPTIMIZER_MAX_STEPS_JSON'] = _json.dumps(optimizer_max_steps_)
            # Also set simple CAP_RANDOM_STEPS for hardcoded fallback in _run_one_optimizer
            for _cap_name in ('RANDOM', 'DE_DIRECT'):
                if _cap_name in optimizer_max_steps_:
                    os.environ['CAP_RANDOM_STEPS'] = str(optimizer_max_steps_[_cap_name])
                    break

        # Suppress verbose competition logging AFTER imports
        logging.disable(logging.CRITICAL)

        # Step A: run only the NEW competitions (competition_start skips already-done ones)
        new_comps_by_synth = [[] for _ in range(current_ns)]
        n_step_a = 0
        if current_nc < TARGET_N_COMP_ and all_synth_30:
            existing_synths = all_synth_30[:current_ns]
            n_new = TARGET_N_COMP_ - current_nc
            n_step_a = current_ns * n_new
            _log(f"{tag} Step A: {current_ns} synths × {n_new} new comps = {n_step_a} competitions")
            with _silence():
                sup, _ = run_harder_competition(
                    dataset_name=dataset, surrogate_type=surrogate, difficulty_config=diff,
                    optimizer_names=OPTIMISERS_, n_competitions=n_new, hidden_fraction=hf,
                    pool_size=200, batch_size=batch, n_jobs=n_jobs_inner, random_state=42,
                    save_results=False, num_of_synthetic_data=current_ns, use_gpu=True,
                    pregenerated_synthetic_datasets=existing_synths, competition_start=current_nc,
                )
            sup_tr = sup.get('tournament_results', [])
            for i in range(current_ns):
                new_comps_by_synth[i] = sup_tr[i].get('competitions', [])
            _log(f"{tag} Step A DONE in {time.time()-t0:.0f}s")

        # Step B: run all 30 competitions on new synthetic datasets
        new_tr_entries = []
        n_step_b = 0
        if current_ns < TARGET_N_SYNTH_ and all_synth_30:
            new_synths = all_synth_30[current_ns:]
            n_step_b = len(new_synths) * TARGET_N_COMP_
            _log(f"{tag} Step B: {len(new_synths)} new synths × {TARGET_N_COMP_} comps = {n_step_b} competitions")
            t_b = time.time()
            with _silence():
                new_res, _ = run_harder_competition(
                    dataset_name=dataset, surrogate_type=surrogate, difficulty_config=diff,
                    optimizer_names=OPTIMISERS_, n_competitions=TARGET_N_COMP_, hidden_fraction=hf,
                    pool_size=200, batch_size=batch, n_jobs=n_jobs_inner, random_state=42,
                    save_results=False, num_of_synthetic_data=len(new_synths), use_gpu=True,
                    pregenerated_synthetic_datasets=new_synths,
                )
            new_tr_entries = new_res.get('tournament_results', [])
            _log(f"{tag} Step B DONE in {time.time()-t_b:.0f}s")

        total = time.time() - t0
        _log(f"{tag} FINISHED: {n_step_a + n_step_b} total comps in {total:.0f}s ({total/max(n_step_a+n_step_b,1):.2f}s/comp)")
        return (hf, batch, new_comps_by_synth, new_tr_entries, total, None)
    except Exception:
        _log(f"{tag} FAILED after {time.time()-t0:.0f}s: {traceback.format_exc()[-200:]}")
        return (hf, batch, None, None, time.time() - t0, traceback.format_exc())


def _run_htl_wave(combo, dataset, wave_jobs, all_synth_30, surrogate, diff,
                  n_jobs, max_workers, optimizer_max_steps, status, save_status,
                  done_count, total_jobs):
    """Run a wave of HTL jobs (subset of pending jobs) via ProcessPoolExecutor.
    Returns updated done_count."""
    if not wave_jobs:
        return done_count

    import os as _os
    n_cpus = _os.cpu_count() or 10
    n_workers = min(len(wave_jobs), max_workers)
    n_jobs_effective = max(1, n_cpus // max(n_workers, 1))
    batch_labels = ','.join(str(b) for _, b, _, _, _, _ in wave_jobs)
    print(f"[{combo}] Launching {n_workers} HTL workers for {dataset} "
          f"(batches=[{batch_labels}], n_jobs_internal={n_jobs_effective}, {n_cpus} CPUs)",
          flush=True)
    t_batch = time.time()

    worker_args = [
        (dataset, hf, batch, surrogate, diff, all_synth_30,
         current_ns, current_nc, n_jobs_effective,
         OPTIMISERS, TARGET_N_SYNTH, TARGET_N_COMP, str(ROOT),
         optimizer_max_steps)
        for (hf, batch, _, _, current_ns, current_nc) in wave_jobs
    ]

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        future_to_job = {
            executor.submit(_htl_worker, wargs): (hf, batch, existing_path, ext_tr, current_ns)
            for wargs, (hf, batch, existing_path, ext_tr, current_ns, _) in zip(worker_args, wave_jobs)
        }
        for future in as_completed(future_to_job):
            hf, batch, existing_path, ext_tr, current_ns = future_to_job[future]
            htl_k = job_key(combo, dataset, hf, 'HTL', batch)
            try:
                res_hf, res_batch, new_comps_by_synth, new_tr_entries, elapsed, err = future.result()
                if err:
                    status[htl_k] = f'failed: {err[:500]}'
                    print(f"[{combo}] HTL FAILED hf={hf} B={batch}: {err}", flush=True)
                else:
                    # Merge Step A results into existing synths
                    for i, new_comps in enumerate(new_comps_by_synth):
                        ext_tr[i]['competitions'].extend(new_comps)
                        ext_tr[i]['n_competitions'] = len(ext_tr[i]['competitions'])
                    # Append Step B new synths
                    ext_tr.extend(new_tr_entries)

                    existing = json.load(open(existing_path))
                    existing['tournament_results'] = ext_tr
                    existing['num_synthetic_datasets'] = len(ext_tr)
                    existing['combined_analysis'] = recompute_combined_analysis_htl(ext_tr)
                    existing['topup_applied'] = True
                    with open(existing_path, 'w') as f:
                        json.dump(existing, f, cls=NumpySafeEncoder)

                    done_count += 1
                    final_nc = len(ext_tr[0].get('competitions', [])) if ext_tr else 0
                    print(f"[{combo}] HTL DONE in {elapsed:.0f}s ({done_count}/{total_jobs}) "
                          f"hf={hf} B={batch} → {len(ext_tr)}s×{final_nc}c", flush=True)
                    status[htl_k] = 'done'
            except Exception as e:
                status[htl_k] = f'failed: {e}'
                print(f"[{combo}] HTL future error hf={hf} B={batch}: {e}", flush=True)
            save_status()

    print(f"[{combo}] HTL wave [{batch_labels}] for {dataset} finished in {time.time()-t_batch:.0f}s",
          flush=True)
    return done_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', required=True, choices=list(COMBO_MAP.keys()))
    parser.add_argument('--n-jobs', type=int, default=2,
                        help='n_jobs passed INSIDE each competition (parallelism within one HTL call).')
    parser.add_argument('--max-workers', type=int, default=3,
                        help='Max parallel ProcessPoolExecutor workers per wave (default 3, was 6).')
    parser.add_argument('--cap-random-steps', type=int, default=60,
                        help='Max steps for RANDOM/DE_DIRECT optimizers (0=unlimited). '
                             'Caps random-walk optimizers that add no value in HTL mode.')
    args = parser.parse_args()

    combo = args.combo
    n_jobs = args.n_jobs          # parallelism inside each competition call
    max_workers = args.max_workers
    surrogate, diff = COMBO_MAP[combo]

    # Build optimizer step caps
    optimizer_max_steps = {}
    if args.cap_random_steps > 0:
        optimizer_max_steps = {
            'RANDOM': args.cap_random_steps,
            'DE_DIRECT': args.cap_random_steps,
        }

    status_file = ROOT / 'New_Results_Official' / f'_topup_status_{combo}.json'
    status = {}
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)

    def save_status():
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    total_htl = len(DATASETS) * len(HIDDEN_FRACTIONS) * len(BATCH_SIZES)
    total_or = len(DATASETS) * len(BATCH_SIZES)
    total_jobs = total_htl + total_or
    done_count = sum(1 for v in status.values() if v == 'done')
    cap_info = f", cap_random={args.cap_random_steps}" if optimizer_max_steps else ""
    print(f"[{combo}] TOP-UP START. {done_count}/{total_jobs} already done. "
          f"max_workers={max_workers}, n_jobs(internal)={n_jobs}{cap_info}", flush=True)
    t_global = time.time()

    for dataset in DATASETS:
        # ── Pre-generate all 30 synthetic datasets once (shared across all HTL jobs) ──
        print(f"[{combo}] Pre-generating 30 synthetics for {dataset}...", flush=True)
        t0 = time.time()
        try:
            with silence_stdout():
                all_synth_30 = pregenerate_synthetic_datasets(
                    dataset, surrogate, diff,
                    pool_size=200, n_synthetic=TARGET_N_SYNTH, random_state=42
                )
            print(f"[{combo}] Synthetics ready in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{combo}] Failed synthetics for {dataset}: {e}", flush=True)
            all_synth_30 = None

        # ── Build list of HTL jobs that still need running ──────────────────────────
        htl_pending = []
        for hf in HIDDEN_FRACTIONS:
            for batch in BATCH_SIZES:
                htl_k = job_key(combo, dataset, hf, 'HTL', batch)
                if status.get(htl_k) == 'done':
                    continue
                existing_path = find_existing_htl_json(combo, dataset, hf, batch)
                if existing_path is None:
                    status[htl_k] = f'failed: No existing HTL JSON for {dataset} hf={hf} B={batch}'
                    save_status()
                    continue
                existing = json.load(open(existing_path))
                ext_tr = existing.get('tournament_results', [])
                current_ns = len(ext_tr)
                current_nc = len(ext_tr[0].get('competitions', [])) if ext_tr else 0
                if current_ns >= TARGET_N_SYNTH and current_nc >= TARGET_N_COMP:
                    print(f"[{combo}] HTL already at target ({current_ns}s×{current_nc}c) hf={hf} B={batch}", flush=True)
                    status[htl_k] = 'done'
                    save_status()
                    continue
                htl_pending.append((hf, batch, existing_path, ext_tr, current_ns, current_nc))

        # ── Run HTL jobs in two waves: batch=1 first (slowest), then batch=10/20 ───
        if htl_pending:
            wave_slow = [j for j in htl_pending if j[1] == 1]   # batch=1 (most steps)
            wave_fast = [j for j in htl_pending if j[1] != 1]   # batch=10, 20

            if wave_slow:
                print(f"[{combo}] === Wave 1: batch=1 jobs ({len(wave_slow)} workers) ===", flush=True)
                done_count = _run_htl_wave(
                    combo, dataset, wave_slow, all_synth_30, surrogate, diff,
                    n_jobs, max_workers, optimizer_max_steps, status, save_status,
                    done_count, total_jobs)

            if wave_fast:
                print(f"[{combo}] === Wave 2: batch=10/20 jobs ({len(wave_fast)} workers) ===", flush=True)
                done_count = _run_htl_wave(
                    combo, dataset, wave_fast, all_synth_30, surrogate, diff,
                    n_jobs, max_workers, optimizer_max_steps, status, save_status,
                    done_count, total_jobs)

        # ── OR top-up (sequential — much faster than HTL) ────────────────────────────
        for batch in BATCH_SIZES:
            or_k_dedup = job_key(combo, dataset, 'or', 'OR', batch)
            # Check all hf-keyed aliases too
            already_done = (status.get(or_k_dedup) == 'done' or
                            any(status.get(job_key(combo, dataset, hf, 'OR', batch)) == 'done'
                                for hf in HIDDEN_FRACTIONS))
            if already_done:
                continue

            print(f"[{combo}] OR  TOPUP B={batch} [{dataset}]", flush=True)
            t_before = time.time()
            try:
                existing_path = find_existing_or_json(combo, dataset, batch)
                if existing_path is None:
                    raise FileNotFoundError(f"No existing OR JSON for {dataset} B={batch}")

                from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition

                with open(existing_path) as f:
                    existing = json.load(f)

                ext_tr = existing.get('tournament_results', {})
                ext_comps = ext_tr.get('competitions', []) if isinstance(ext_tr, dict) else []
                current_nc = len(ext_comps)

                if current_nc >= TARGET_N_COMP:
                    print(f"[{combo}] OR already at target ({current_nc}c)", flush=True)
                    status[or_k_dedup] = 'done'
                    save_status()
                    continue

                with silence_stdout():
                    sup_result = run_harder_open_race_competition(
                        dataset_name=dataset, surrogate_type=surrogate, difficulty_config=diff,
                        optimizer_names=OPTIMISERS, n_competitions=TARGET_N_COMP,
                        S=200, R=10, B=batch, n_jobs=-1, random_state=42, save_results=False,
                    )

                sup_tr = sup_result[0] if isinstance(sup_result, tuple) else sup_result
                new_comps = sup_tr.get('competitions', []) if isinstance(sup_tr, dict) else []

                ext_comps.extend(new_comps[current_nc:])
                if isinstance(ext_tr, dict):
                    ext_tr['competitions'] = ext_comps
                    ext_tr['n_competitions'] = len(ext_comps)
                    if new_comps:
                        ext_tr['optimizer_names'] = list(new_comps[0].get('optimizer_results', {}).keys())
                existing['tournament_results'] = ext_tr
                existing['analysis'] = recompute_analysis_or(ext_tr)
                existing['topup_applied'] = True

                with open(existing_path, 'w') as f:
                    json.dump(existing, f, cls=NumpySafeEncoder)

                elapsed = time.time() - t_before
                done_count += 1
                print(f"[{combo}] OR  DONE in {elapsed:.0f}s ({done_count}/{total_jobs}) → {len(ext_comps)}c", flush=True)
                status[or_k_dedup] = 'done'
            except Exception as e:
                status[or_k_dedup] = f'failed: {e}'
                print(f"[{combo}] OR  FAILED: {e}", flush=True)
                traceback.print_exc()
            save_status()

        all_synth_30 = None  # free memory before next dataset

    total_elapsed = time.time() - t_global
    done_count = sum(1 for v in status.values() if v == 'done')
    fail_count = sum(1 for v in status.values() if 'failed' in str(v))
    print(f"[{combo}] TOP-UP FINISHED in {total_elapsed/60:.1f} min. {done_count}/{total_jobs} done, {fail_count} failed.", flush=True)


if __name__ == '__main__':
    main()
