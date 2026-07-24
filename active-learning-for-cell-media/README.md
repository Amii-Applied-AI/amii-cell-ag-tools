# A data-driven evaluation of optimization techniques in cell culture media

Code and input data for the *Bioinformatics* paper **"A data-driven evaluation of
optimization techniques in cell culture media."** It benchmarks Bayesian
optimization, design-of-experiments, and baseline search strategies on six
published cell-culture datasets, under two protocols and two difficulty regimes.

## Contents

| Path | Description |
|---|---|
| `Data/` | The six cell-culture / cell-media datasets, reformatted to a common schema. Original sources and licensing are in `Data/SOURCES.md`. |
| `utils/` | Dataset registry, optimizer implementations, surrogate models, and sparse-GP approximations. |
| `experiments/` | The two benchmark protocols (Hide-the-Label, Open Race) under Regular and Hard difficulty, plus figure and statistical-analysis scripts. |
| `reproduce/` | Scripts that regenerate the paper's benchmark results. |
| `landing_page/` | Static interactive playground for exploring the benchmark results in a browser. |
| `main.py` | CLI for running a single ad-hoc experiment. |

Results are not bundled; they are regenerated with the scripts in `reproduce/`.

## Interactive playground

`landing_page/` is a static site for comparing optimizers across surrogate models,
difficulty modes, batch sizes, hidden fractions, and datasets. Open
`landing_page/index.html` in a browser, or view the hosted version linked from the
paper. It reads precomputed summaries in `landing_page/playground_data.js`; no server
or build step is required.

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch is required only for the Bayesian neural-network surrogate; the Gaussian
process, random forest, and neural-network surrogates depend only on scikit-learn.

## Benchmark design

Each run pairs a **surrogate model** (which defines the ground-truth landscape) with
a **difficulty regime**, giving eight combinations:

| Surrogate | Regular | Hard |
|---|---|---|
| Gaussian process (GP) | `GP_Regular` | `GP_Hard` |
| Random forest (RF) | `RF_Regular` | `RF_Hard` |
| Neural network (NN) | `NN_Regular` | `NN_Hard` |
| Bayesian neural network (BNN) | `BNN_Regular` | `BNN_Hard` |

Each combination is evaluated on all six datasets under two protocols:

- **Hide-the-Label** — the optimizer sequentially reveals points from a hidden pool;
  the metric is the number of steps to reach the target. Run at hidden fractions
  0.95 and 0.99 and batch sizes 1, 10, and 20.
- **Open Race** — the optimizer has a fixed budget of 200 evaluations from 10
  initial points; the metric is the best value found so far.

The 14-optimizer pool is: `BO_GP_EI`, `SBO_GP_PV`, `SBO_GP_EI_TRUNCDE`,
`SBO_POLY_PV`, `SBO_ANN_PV`, `SMART_BO` (Bayesian / surrogate-assisted);
`FULL_FACTORIAL`, `FRACTIONAL_FACTORIAL`, `PLACKETT_BURMAN`, `CENTRAL_COMPOSITE`,
`BOX_BEHNKEN`, `LATIN_HYPERCUBE` (design of experiments); and `DE_DIRECT`, `RANDOM`
(baselines). Implementations are in `utils/optimizers.py`.

## Reproducing the results

All runs use `random_state=42` and 3 competitions x 10 synthetic datasets (Hide-the-Label)
or 10 competitions (Open Race) per cell.

```bash
# Hide-the-Label, hidden fraction 0.95, all batch sizes
python reproduce/run_hide_the_label.py --hf 0.95 --batch 1
python reproduce/run_hide_the_label.py --hf 0.95 --batch 10
python reproduce/run_hide_the_label.py --hf 0.95 --batch 20

# Hide-the-Label at the sparse initialization (0.99); GP and RF only
python reproduce/run_hide_the_label.py --hf 0.99 --batch 1 --combos GP_Regular GP_Hard RF_Regular RF_Hard

# Open Race (no hidden-fraction dependence)
python reproduce/run_open_race.py --batch 1
python reproduce/run_open_race.py --batch 10
python reproduce/run_open_race.py --batch 20
```

Each script is resumable (it skips cells whose output already exists) and writes to
`results/`. A full sweep is CPU-bound; to use all cores, run one cell per process in
parallel, for example:

```bash
for combo in GP_Regular GP_Hard RF_Regular RF_Hard NN_Regular NN_Hard BNN_Regular BNN_Hard; do
  for ds in DBO_dataset_rat_myocyte MOBO_dataset_rat_myocyte df_Human_Hela_regular_mode \
            df_Human_Hela_timesaving_mode df_Human_T_Cell_Expanded df_Human_TF_Cell_Expanded; do
    echo "--combos $combo --datasets $ds"
  done
done | xargs -P "$(nproc)" -L1 python reproduce/run_hide_the_label.py --hf 0.95 --batch 1
```

Per-pool seeding is deterministic per cell (`synthetic_dataset_index = 0 .. n_synth-1`),
so a cell's result is the same whether it runs on its own or alongside others.

A single ad-hoc experiment can also be run through the CLI:

```bash
python main.py hard-open df_Human_TF_Cell_Expanded --surrogate_type random_forest --S 200 --R 10 --B 1
```

## Datasets

Passed by name on the command line (registry in `utils/datasets.py`):
`DBO_dataset_rat_myocyte`, `MOBO_dataset_rat_myocyte`, `df_Human_Hela_regular_mode`,
`df_Human_Hela_timesaving_mode`, `df_Human_T_Cell_Expanded`, `df_Human_TF_Cell_Expanded`.
Original publications and licensing are listed in `Data/SOURCES.md`.

## Implementation notes

- **Gaussian process surrogate.** The GP kernel uses a per-dimension (ARD) length scale
  initialized from the median pairwise distance of the standardized inputs, with bounded
  length scales and a bounded white-noise term. This gives stable, well-conditioned fits
  across the range of input dimensionalities spanned by the datasets
  (`utils/harder_surrogates.py`).
- **Bayesian neural network.** The MC-Dropout surrogate seeds the torch RNG before each
  fit and prediction and runs single-threaded, so its stochastic forward passes are fully
  reproducible.
- **Determinism.** All experiments use a fixed random seed (42) and deterministic
  per-cell pool seeding, so results are reproducible across machines and independent of
  the order in which cells are run.

## Hardware

A GPU is only needed for the Bayesian neural-network surrogate; everything else runs on
CPU. A full sweep across all combinations takes on the order of a day on a
many-core machine at one cell per process.

## Citation

If you use this code, please cite both the software and the paper (see `CITATION.cff`).
An archived release is available on Zenodo: [10.5281/zenodo.21501659](https://doi.org/10.5281/zenodo.21501659).
The datasets are derived from previously published work; please also cite the original
sources in `Data/SOURCES.md`. Released under the Creative Commons Attribution 4.0 International License (`LICENSE`).
