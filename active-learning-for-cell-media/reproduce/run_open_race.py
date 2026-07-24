"""Regenerate the Open Race benchmark results.

Runs the eight (surrogate x difficulty) combinations across the six datasets.
Each optimizer gets a fixed budget of S=200 evaluations starting from R=10 random
points; the recorded quantity is the best objective value found so far. Uses the
14-optimizer pool, 10 competitions per cell, and seed 42.

Open Race does not depend on a hidden fraction (the budget and initial-point count
are fixed), so there is a single set of results per (surrogate, mode, batch).

Examples:
    python reproduce/run_open_race.py --batch 1
    python reproduce/run_open_race.py --batch 10 --combos GP_Regular RF_Hard
"""
import sys, os, io, json, time, argparse, contextlib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.INFO)
from experiments.Hard_Mode.Harder_Mode_Open_Race import run_harder_open_race_competition

COMBOS = [
    ("GP",  "Regular", "gaussian_process"),        ("GP",  "Hard", "gaussian_process"),
    ("RF",  "Regular", "random_forest"),           ("RF",  "Hard", "random_forest"),
    ("NN",  "Regular", "neural_network"),          ("NN",  "Hard", "neural_network"),
    ("BNN", "Regular", "bayesian_neural_network"), ("BNN", "Hard", "bayesian_neural_network"),
]
REGULAR = dict(noise_level=0.0, heteroscedastic=False, n_modes=1, mode_separation=2.0)
NOISE   = dict(noise_level=0.1, heteroscedastic=False, n_modes=1, mode_separation=2.0)
BIMODAL = dict(noise_level=0.2, heteroscedastic=True,  n_modes=2, mode_separation=2.0)
HARD_BY_DATASET = {
    "DBO_dataset_rat_myocyte": NOISE, "MOBO_dataset_rat_myocyte": NOISE,
    "df_Human_Hela_regular_mode": NOISE, "df_Human_Hela_timesaving_mode": NOISE,
    "df_Human_T_Cell_Expanded": BIMODAL, "df_Human_TF_Cell_Expanded": BIMODAL,
}
DATASETS = list(HARD_BY_DATASET)
OPTIMIZERS = ["RANDOM", "BO_GP_EI", "SBO_GP_PV", "SBO_GP_EI_TRUNCDE", "SBO_POLY_PV",
              "SBO_ANN_PV", "SMART_BO", "FULL_FACTORIAL", "FRACTIONAL_FACTORIAL",
              "LATIN_HYPERCUBE", "CENTRAL_COMPOSITE", "BOX_BEHNKEN", "PLACKETT_BURMAN", "DE_DIRECT"]


def numpy_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (set, tuple)): return list(o)
    try: return float(o)
    except Exception: return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-comp", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--combos", nargs="*", default=[f"{s}_{m}" for s, m, _ in COMBOS])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, "results", "open_race", f"b{args.batch}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    for surrogate, mode, surrogate_type in COMBOS:
        combo = f"{surrogate}_{mode}"
        if combo not in args.combos:
            continue
        combo_dir = os.path.join(out_dir, combo)
        os.makedirs(combo_dir, exist_ok=True)
        for dataset in args.datasets:
            out_path = os.path.join(combo_dir, f"{dataset}.json")
            if os.path.exists(out_path):
                continue
            config = REGULAR if mode == "Regular" else HARD_BY_DATASET[dataset]
            print(f"[{time.strftime('%H:%M:%S')}] {combo:12s} {dataset}", flush=True)
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                results, analysis = run_harder_open_race_competition(
                    dataset_name=dataset, surrogate_type=surrogate_type, difficulty_config=config,
                    optimizer_names=OPTIMIZERS, n_competitions=args.n_comp, S=200, R=10,
                    B=args.batch, n_jobs=1, random_state=42, save_results=False, use_cache_model=False)
            record = {"dataset": dataset, "surrogate": surrogate, "mode": mode, "config": config,
                      "batch": args.batch, "S": 200, "R": 10, "n_competitions": args.n_comp,
                      "tournament_results": results, "analysis": analysis}
            json.dump(record, open(out_path, "w"), default=numpy_default)
            print(f"           done in {time.time() - t:.0f}s", flush=True)

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
