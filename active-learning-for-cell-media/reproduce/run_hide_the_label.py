"""Regenerate the Hide-the-Label benchmark results.

Runs the eight (surrogate x difficulty) combinations across the six datasets and
writes, for every cell, the full tournament results plus a mean-steps summary.
Configuration matches the published results: the 14-optimizer pool, 3 competitions
x 10 synthetic datasets per cell, a candidate pool of 200 points, and seed 42.

Per-pool seeding is deterministic per cell (synthetic_dataset_index = 0..n_synth-1),
so results do not depend on how many cells share a process. For large sweeps, run
one cell per process (--combos/--datasets) in parallel; each process is independent.

Examples:
    python reproduce/run_hide_the_label.py --batch 1  --hf 0.95
    python reproduce/run_hide_the_label.py --batch 10 --hf 0.95
    python reproduce/run_hide_the_label.py --batch 1  --hf 0.99 --combos GP_Regular RF_Hard
"""
import sys, os, io, json, time, argparse, contextlib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.INFO)
from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import run_harder_competition

COMBOS = [
    ("GP",  "Regular", "gaussian_process"),        ("GP",  "Hard", "gaussian_process"),
    ("RF",  "Regular", "random_forest"),           ("RF",  "Hard", "random_forest"),
    ("NN",  "Regular", "neural_network"),          ("NN",  "Hard", "neural_network"),
    ("BNN", "Regular", "bayesian_neural_network"), ("BNN", "Hard", "bayesian_neural_network"),
]

# Difficulty configurations. Regular is noise-free; Hard adds noise (and, for the
# two immune-cell datasets, bimodality) on top of the surrogate landscape.
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
    ap.add_argument("--n-comp", type=int, default=3)
    ap.add_argument("--n-synth", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hf", type=float, default=0.95, help="hidden fraction (0.95 or 0.99)")
    ap.add_argument("--combos", nargs="*", default=[f"{s}_{m}" for s, m, _ in COMBOS])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default=None, help="output directory (default: results/hide_the_label/...)")
    args = ap.parse_args()

    tag = f"hf{int(args.hf * 100)}_b{args.batch}"
    out_dir = args.out or os.path.join(ROOT, "results", "hide_the_label", tag)
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
                results, analysis = run_harder_competition(
                    dataset_name=dataset, surrogate_type=surrogate_type, difficulty_config=config,
                    optimizer_names=OPTIMIZERS, n_competitions=args.n_comp, hidden_fraction=args.hf,
                    pool_size=200, batch_size=args.batch, n_jobs=1, random_state=42,
                    save_results=False, num_of_synthetic_data=args.n_synth,
                    use_gpu=True, pregenerated_synthetic_datasets=None)
            record = {"dataset": dataset, "surrogate": surrogate, "mode": mode, "config": config,
                      "batch": args.batch, "hidden_fraction": args.hf,
                      "n_competitions": args.n_comp, "n_synth": args.n_synth,
                      "tournament_results": results, "analysis": analysis}
            json.dump(record, open(out_path, "w"), default=numpy_default)
            stats = analysis.get("optimizer_stats", {})
            summary = {o: {"mean_steps": float(stats[o]["mean_steps"]),
                           "std_steps": float(stats[o].get("std_steps", 0.0))} for o in stats}
            json.dump({"dataset": dataset, "surrogate": surrogate, "mode": mode,
                       "batch": args.batch, "hidden_fraction": args.hf, "mean_steps": summary},
                      open(out_path.replace(".json", "_summary.json"), "w"))
            print(f"           done in {time.time() - t:.0f}s", flush=True)

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
