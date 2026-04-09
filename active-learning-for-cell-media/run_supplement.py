"""
run_supplement.py
=================
Runs ONLY SBO_ANN_PV and SMART_BO for all combos/datasets/settings,
then merges their results into the existing 12-optimizer JSONs to produce
complete 14-optimizer result files.

Usage:
    python3 run_supplement.py --combo GP_Hard --n-jobs 2
"""

import sys, os, time, logging, json, shutil, argparse, glob
from pathlib import Path
from contextlib import contextmanager

logging.disable(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

@contextmanager
def silence_stdout():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition, HarderHideLabelCompetition
from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition
from utils.datasets import load_dataset
import numpy as np

HTL_RESULTS_DIR = ROOT / 'Results' / 'Hard_Mode' / 'Hide_The_Label'
OR_RESULTS_DIR  = ROOT / 'Results' / 'Hard_Mode' / 'Open_Race'
HTL_PLOT_DIR    = ROOT / 'Plotting' / 'Hard_Mode' / 'Hide_The_Label'
OR_PLOT_DIR     = ROOT / 'Plotting' / 'Hard_Mode' / 'Open_Race'

DATASETS = [
    'MOBO_dataset_rat_myocyte',
    'DBO_dataset_rat_myocyte',
    'df_Human_Hela_regular_mode',
    'df_Human_Hela_timesaving_mode',
    'df_Human_T_Cell_Expanded',
    'df_Human_TF_Cell_Expanded',
]

SUPPLEMENT_OPTIMISERS = ['SBO_ANN_PV', 'SMART_BO']

NO_DIFF = {'noise_level': 0.0, 'heteroscedastic': False, 'n_modes': 1, 'mode_separation': 2.0}
HARD    = {'noise_level': 0.1, 'heteroscedastic': True,  'n_modes': 2, 'mode_separation': 2.0}

COMBO_MAP = {
    'GP_Regular': ('gaussian_process', NO_DIFF),
    'GP_Hard':    ('gaussian_process', HARD),
    'RF_Regular': ('random_forest',    NO_DIFF),
    'RF_Hard':    ('random_forest',    HARD),
    'NN_Regular': ('neural_network',   NO_DIFF),
    'NN_Hard':    ('neural_network',   HARD),
}

HIDDEN_FRACTIONS = [0.95, 0.99]
BATCH_SIZES      = [1, 10, 20]
N_COMPETITIONS   = 10
N_SYNTHETIC      = 10


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
                                   pool_size=200, n_synthetic=10, random_state=42):
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
    synthetic_datasets = []
    for synth_idx in range(n_synthetic):
        synthetic_seed = random_state + synth_idx * 1000
        X_pool, y_pool = competition.generate_candidate_pool(
            surrogate_model, scaler, X, n_points=pool_size, synthetic_seed=synthetic_seed
        )
        synthetic_datasets.append((X_pool, y_pool))
    return synthetic_datasets


def infer_batch_size(competition):
    """Infer batch_size from optimization_history of any optimizer result."""
    for opt_result in competition.get('optimizer_results', {}).values():
        history = opt_result.get('optimization_history', [])
        if history:
            max_bp = max(h.get('batch_position', 0) for h in history if h.get('step', -1) == 0)
            return max_bp + 1
    return None


def find_existing_htl_json(dataset_name, surrogate_type, difficulty_config,
                            hidden_fraction, batch_size, n_competitions, n_synthetic):
    """Find the most recent 12-optimizer HTL JSON matching these exact settings."""
    candidates = []
    for f in sorted(HTL_RESULTS_DIR.glob(f"{dataset_name}_hard_mode_hide_the_label_*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            # Check top-level fields
            if d.get('surrogate_type') != surrogate_type:
                continue
            if d.get('difficulty_config') != difficulty_config:
                continue
            if d.get('num_synthetic_datasets') != n_synthetic:
                continue
            tr = d.get('tournament_results', [])
            if not tr or len(tr) != n_synthetic:
                continue
            t0 = tr[0]
            if t0.get('n_competitions') != n_competitions:
                continue
            if abs(t0.get('hidden_fraction', 0) - hidden_fraction) > 0.001:
                continue
            # Check batch_size
            comps = t0.get('competitions', [])
            if not comps:
                continue
            bs = infer_batch_size(comps[0])
            if bs != batch_size:
                continue
            # Check it has the 12 optimizers (not already 14)
            opt_names = list(comps[0].get('optimizer_results', {}).keys())
            if 'SMART_BO' in opt_names or 'SBO_ANN_PV' in opt_names:
                continue  # already has them
            candidates.append(f)
        except Exception:
            continue
    return candidates[-1] if candidates else None  # most recent


def find_existing_or_json(dataset_name, surrogate_type, difficulty_config,
                           batch_size, n_competitions):
    """Find the most recent 12-optimizer OR JSON matching these settings."""
    candidates = []
    for f in sorted(OR_RESULTS_DIR.glob(f"{dataset_name}_hard_mode_open_race_*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            if d.get('surrogate_type') != surrogate_type:
                continue
            if d.get('difficulty_config') != difficulty_config:
                continue
            # OR tournament_results is a dict (not a list)
            tr = d.get('tournament_results', {})
            t0 = tr if isinstance(tr, dict) else (tr[0] if isinstance(tr, list) and tr else None)
            if t0 is None:
                continue
            if t0.get('n_competitions') != n_competitions:
                continue
            # B is stored directly in the tournament dict
            if t0.get('B') != batch_size:
                continue
            comps = t0.get('competitions', [])
            if not comps:
                continue
            opt_names = list(comps[0].get('optimizer_results', {}).keys())
            if 'SMART_BO' in opt_names or 'SBO_ANN_PV' in opt_names:
                continue
            candidates.append(f)
        except Exception:
            continue
    return candidates[-1] if candidates else None


def recompute_combined_analysis(tournament_results):
    """Recompute combined_analysis from tournament_results after merge."""
    all_opt_names = list(tournament_results[0]['competitions'][0]['optimizer_results'].keys())
    stats = {name: {'steps_list': [], 'found_list': []} for name in all_opt_names}

    for tr in tournament_results:
        for comp in tr.get('competitions', []):
            for name, res in comp.get('optimizer_results', {}).items():
                if name in stats:
                    stats[name]['steps_list'].append(res.get('steps_to_target', None))
                    stats[name]['found_list'].append(res.get('found_target', False))

    optimizer_stats = {}
    for name, s in stats.items():
        steps = [x for x in s['steps_list'] if x is not None]
        found = s['found_list']
        optimizer_stats[name] = {
            'mean_steps': float(np.mean(steps)) if steps else None,
            'std_steps': float(np.std(steps)) if steps else None,
            'median_steps': float(np.median(steps)) if steps else None,
            'success_rate': float(np.mean(found)) if found else 0.0,
            'n_competitions': len(found),
        }

    # Rank by mean_steps (lower is better)
    ranked = sorted(
        optimizer_stats.keys(),
        key=lambda n: optimizer_stats[n]['mean_steps'] if optimizer_stats[n]['mean_steps'] is not None else 1e9
    )

    return {
        'optimizer_names': all_opt_names,
        'optimizer_stats': optimizer_stats,
        'ranking': ranked,
        'n_competitions': sum(len(tr.get('competitions', [])) for tr in tournament_results),
        'dataset_name': tournament_results[0].get('dataset_name'),
        'hidden_fraction': tournament_results[0].get('hidden_fraction'),
        'pool_size': tournament_results[0].get('pool_size'),
    }


def merge_htl_results(existing_path, supplement_result):
    """
    Merge supplement (2-optimizer) results into existing (12-optimizer) JSON.
    supplement_result is the dict returned by run_harder_competition.
    Returns the merged dict.
    """
    with open(existing_path) as f:
        existing = json.load(f)

    sup_tr = supplement_result.get('tournament_results', [])
    ext_tr = existing.get('tournament_results', [])

    if len(sup_tr) != len(ext_tr):
        raise ValueError(f"Synthetic run count mismatch: {len(sup_tr)} vs {len(ext_tr)}")

    for i, (sup_run, ext_run) in enumerate(zip(sup_tr, ext_tr)):
        sup_comps = sup_run.get('competitions', [])
        ext_comps = ext_run.get('competitions', [])
        if len(sup_comps) != len(ext_comps):
            raise ValueError(f"Competition count mismatch in synthetic run {i}: {len(sup_comps)} vs {len(ext_comps)}")

        for j, (sup_comp, ext_comp) in enumerate(zip(sup_comps, ext_comps)):
            # Merge optimizer_results
            for opt_name, opt_res in sup_comp.get('optimizer_results', {}).items():
                ext_comp['optimizer_results'][opt_name] = opt_res

        # Update optimizer_names in this tournament run
        all_names = list(ext_run['competitions'][0]['optimizer_results'].keys())
        ext_run['optimizer_names'] = all_names

    # Recompute combined_analysis
    existing['combined_analysis'] = recompute_combined_analysis(existing['tournament_results'])

    # Mark as having supplement optimizers
    existing['supplement_merged'] = True
    existing['supplement_optimizers'] = SUPPLEMENT_OPTIMISERS

    return existing


def save_merged_json(merged, original_path, combo, hf, batch, protocol):
    """Save merged JSON to New_Results_Official/<combo>/complete/ directory."""
    dest_dir = ROOT / 'New_Results_Official' / combo / f'hiddenfrac{int(hf*100)}' / protocol / 'complete'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / original_path.name
    with open(dest_file, 'w') as f:
        json.dump(merged, f)
    return dest_file


def job_key(combo, dataset, hf, protocol, batch):
    return f"{combo}|{dataset}|hf{hf}|{protocol}|B{batch}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', required=True, choices=list(COMBO_MAP.keys()))
    parser.add_argument('--n-jobs', type=int, default=2)
    args = parser.parse_args()

    combo = args.combo
    n_jobs = args.n_jobs
    surrogate, diff = COMBO_MAP[combo]

    status_file = ROOT / 'New_Results_Official' / f'_supplement_status_{combo}.json'
    status = {}
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)

    def save_status():
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    total_jobs = len(DATASETS) * len(HIDDEN_FRACTIONS) * len(BATCH_SIZES) * 2
    done_count = sum(1 for v in status.values() if v == 'done')
    print(f"[{combo}] SUPPLEMENT START. {done_count}/{total_jobs} already done. n_jobs={n_jobs}", flush=True)
    t_start = time.time()

    for dataset in DATASETS:
        # Pre-generate synthetics once per dataset (HTL only)
        htl_needs_run = any(
            status.get(job_key(combo, dataset, hf, 'HTL', batch)) != 'done'
            for hf in HIDDEN_FRACTIONS for batch in BATCH_SIZES
        )
        synth_data = None
        if htl_needs_run:
            print(f"[{combo}] Pre-generating synthetics for {dataset}...", flush=True)
            t0 = time.time()
            try:
                with silence_stdout():
                    synth_data = pregenerate_synthetic_datasets(
                        dataset, surrogate, diff,
                        pool_size=200, n_synthetic=N_SYNTHETIC, random_state=42,
                    )
                print(f"[{combo}] Synthetics ready in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[{combo}] Failed synthetics: {e}", flush=True)
                synth_data = None

        for hf in HIDDEN_FRACTIONS:
            hf_tag = f"hiddenfrac{int(hf*100)}"

            for batch in BATCH_SIZES:
                # ── HTL supplement ──────────────────────────────────────────
                htl_k = job_key(combo, dataset, hf, 'HTL', batch)
                if status.get(htl_k) != 'done':
                    print(f"[{combo}] HTL hf={hf} B={batch} [{dataset}]", flush=True)
                    t_before = time.time()
                    try:
                        # Find existing 12-optimizer JSON to merge into
                        existing_path = find_existing_htl_json(
                            dataset, surrogate, diff, hf, batch, N_COMPETITIONS, N_SYNTHETIC
                        )
                        if existing_path is None:
                            raise FileNotFoundError(
                                f"No existing 12-opt HTL JSON for {dataset} hf={hf} B={batch}"
                            )

                        # Run ONLY the 2 supplement optimizers
                        with silence_stdout():
                            sup_result, _ = run_harder_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=SUPPLEMENT_OPTIMISERS,
                                n_competitions=N_COMPETITIONS,
                                hidden_fraction=hf,
                                pool_size=200,
                                batch_size=batch,
                                n_jobs=n_jobs,
                                random_state=42,
                                save_results=False,   # don't save — we'll save merged
                                num_of_synthetic_data=N_SYNTHETIC,
                                use_gpu=True,
                                pregenerated_synthetic_datasets=synth_data,
                            )

                        # Merge and save
                        merged = merge_htl_results(existing_path, sup_result)
                        out_path = save_merged_json(merged, existing_path, combo, hf, batch, 'Hide_The_Label')

                        elapsed = time.time() - t_before
                        done_count += 1
                        print(f"[{combo}] HTL DONE in {elapsed:.0f}s ({done_count}/{total_jobs}) → {out_path.name}", flush=True)
                        status[htl_k] = 'done'
                    except Exception as e:
                        import traceback
                        status[htl_k] = f'failed: {e}'
                        print(f"[{combo}] HTL FAILED: {e}", flush=True)
                        traceback.print_exc()
                    save_status()

                # ── OR supplement ───────────────────────────────────────────
                # OR has no hidden_fraction — use a batch-level key, run once per batch
                or_k = job_key(combo, dataset, 'or', 'OR', batch)
                or_hf_k = job_key(combo, dataset, hf, 'OR', batch)  # legacy key in status
                if status.get(or_k) != 'done' and status.get(or_hf_k) != 'done':
                    print(f"[{combo}] OR  B={batch} [{dataset}]", flush=True)
                    t_before = time.time()
                    try:
                        existing_path = find_existing_or_json(
                            dataset, surrogate, diff, batch, N_COMPETITIONS
                        )
                        if existing_path is None:
                            raise FileNotFoundError(
                                f"No existing 12-opt OR JSON for {dataset} B={batch}"
                            )

                        with silence_stdout():
                            sup_result_tuple = run_harder_open_race_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=SUPPLEMENT_OPTIMISERS,
                                n_competitions=N_COMPETITIONS,
                                S=200,
                                R=10,
                                B=batch,
                                n_jobs=n_jobs,
                                random_state=42,
                                save_results=False,
                            )
                        # Returns (tournament_results, analysis) tuple
                        sup_tr_dict = sup_result_tuple[0] if isinstance(sup_result_tuple, tuple) else sup_result_tuple

                        # OR merge — tournament_results is a dict, competitions is a list inside
                        with open(existing_path) as f:
                            existing_or = json.load(f)

                        ext_tr = existing_or.get('tournament_results', {})
                        sup_comps = sup_tr_dict.get('competitions', []) if isinstance(sup_tr_dict, dict) else []
                        ext_comps = ext_tr.get('competitions', []) if isinstance(ext_tr, dict) else []

                        for sup_comp, ext_comp in zip(sup_comps, ext_comps):
                            for opt_name, opt_res in sup_comp.get('optimizer_results', {}).items():
                                ext_comp['optimizer_results'][opt_name] = opt_res

                        if ext_comps:
                            ext_tr['optimizer_names'] = list(ext_comps[0]['optimizer_results'].keys())

                        existing_or['supplement_merged'] = True
                        existing_or['supplement_optimizers'] = SUPPLEMENT_OPTIMISERS

                        out_dir = ROOT / 'New_Results_Official' / combo / 'Open_Race' / 'complete'
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / existing_path.name
                        with open(out_path, 'w') as f:
                            json.dump(existing_or, f)

                        elapsed = time.time() - t_before
                        done_count += 1
                        print(f"[{combo}] OR  DONE in {elapsed:.0f}s ({done_count}/{total_jobs}) → {out_path.name}", flush=True)
                        status[or_k] = 'done'
                    except Exception as e:
                        import traceback
                        status[or_k] = f'failed: {e}'
                        print(f"[{combo}] OR  FAILED: {e}", flush=True)
                        traceback.print_exc()
                    save_status()

        synth_data = None

    total_elapsed = time.time() - t_start
    done_count = sum(1 for v in status.values() if v == 'done')
    fail_count = sum(1 for v in status.values() if 'failed' in str(v))
    print(f"[{combo}] SUPPLEMENT FINISHED in {total_elapsed/60:.1f} min. {done_count}/{total_jobs} done, {fail_count} failed.", flush=True)


if __name__ == '__main__':
    main()
