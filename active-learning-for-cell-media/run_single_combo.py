"""
run_single_combo.py
===================
Runs a single combo (e.g., GP_Hard) with all its 72 jobs.
Designed to be launched in parallel by run_parallel.py.
"""

import sys, os, time, logging, json, shutil, argparse
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Suppress library-level INFO spam, keep WARNING+
logging.disable(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

@contextmanager
def silence_stdout():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
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

HTL_PLOT_DIR = ROOT / 'Plotting' / 'Hard_Mode' / 'Hide_The_Label'
OR_PLOT_DIR  = ROOT / 'Plotting' / 'Hard_Mode' / 'Open_Race'

DATASETS = [
    'MOBO_dataset_rat_myocyte',
    'DBO_dataset_rat_myocyte',
    'df_Human_Hela_regular_mode',
    'df_Human_Hela_timesaving_mode',
    'df_Human_T_Cell_Expanded',
    'df_Human_TF_Cell_Expanded',
]

OPTIMISERS = [
    # 12-optimizer base pool. SBO_ANN_PV and SMART_BO are added by
    # run_supplement.py as a separate merge step (matches the layout of
    # the canonical files in New_Results_Official/, which have
    # `supplement_merged: True`).
    'RANDOM',
    'BO_GP_EI', 'SBO_GP_PV', 'SBO_GP_EI_TRUNCDE',
    'SBO_POLY_PV',
    'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'LATIN_HYPERCUBE',
    'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'PLACKETT_BURMAN',
    'DE_DIRECT',
]

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

import numpy as np


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
                                   pool_size=200, n_synthetic=30, random_state=42):
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

    status_file = ROOT / "New_Results_Official" / f"_run_status_{combo}.json"
    out_base = ROOT / "New_Results_Official" / combo

    # Load existing status for resume
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
    print(f"[{combo}] Starting. {done_count}/{total_jobs} already done. n_jobs={n_jobs}", flush=True)
    t_start = time.time()

    for ds_idx, dataset in enumerate(DATASETS):
        # Check if any HTL job for this dataset needs running
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
                # HTL
                htl_k = job_key(combo, dataset, hf, 'HTL', batch)
                if status.get(htl_k) != 'done':
                    print(f"[{combo}] HTL hf={hf} B={batch} [{dataset}]", flush=True)
                    htl_dest = out_base / hf_tag / 'Hide_The_Label'
                    t_before = time.time()
                    try:
                        with silence_stdout():
                            run_harder_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=OPTIMISERS,
                                n_competitions=N_COMPETITIONS,
                                hidden_fraction=hf,
                                pool_size=200,
                                batch_size=batch,
                                n_jobs=n_jobs,
                                random_state=42,
                                save_results=True,
                                num_of_synthetic_data=N_SYNTHETIC,
                                use_gpu=True,
                                pregenerated_synthetic_datasets=synth_data,
                            )
                        n_copied = copy_new_plots(HTL_PLOT_DIR, htl_dest, t_before)
                        elapsed = time.time() - t_before
                        done_count += 1
                        print(f"[{combo}] HTL DONE in {elapsed:.0f}s ({done_count}/{total_jobs}) — {n_copied} plots", flush=True)
                        status[htl_k] = 'done'
                    except Exception as e:
                        status[htl_k] = f'failed: {e}'
                        print(f"[{combo}] HTL FAILED: {e}", flush=True)
                    save_status()

                # Open Race
                or_k = job_key(combo, dataset, hf, 'OR', batch)
                if status.get(or_k) != 'done':
                    print(f"[{combo}] OR  hf={hf} B={batch} [{dataset}]", flush=True)
                    or_dest = out_base / hf_tag / 'Open_Race'
                    t_before = time.time()
                    try:
                        with silence_stdout():
                            run_harder_open_race_competition(
                                dataset_name=dataset,
                                surrogate_type=surrogate,
                                difficulty_config=diff,
                                optimizer_names=OPTIMISERS,
                                n_competitions=N_COMPETITIONS,
                                S=200,
                                R=10,
                                B=batch,
                                n_jobs=n_jobs,
                                random_state=42,
                                save_results=True,
                            )
                        n_copied = copy_new_plots(OR_PLOT_DIR, or_dest, t_before)
                        elapsed = time.time() - t_before
                        done_count += 1
                        print(f"[{combo}] OR  DONE in {elapsed:.0f}s ({done_count}/{total_jobs}) — {n_copied} plots", flush=True)
                        status[or_k] = 'done'
                    except Exception as e:
                        status[or_k] = f'failed: {e}'
                        print(f"[{combo}] OR  FAILED: {e}", flush=True)
                    save_status()

        synth_data = None  # free memory

    total_elapsed = time.time() - t_start
    done_count = sum(1 for v in status.values() if v == 'done')
    fail_count = sum(1 for v in status.values() if 'failed' in str(v))
    print(f"[{combo}] FINISHED in {total_elapsed/60:.1f} min. {done_count}/{total_jobs} done, {fail_count} failed.", flush=True)


if __name__ == '__main__':
    main()
