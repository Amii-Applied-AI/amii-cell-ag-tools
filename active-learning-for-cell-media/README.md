# Active-Learning Surrogate Competition — Code Package

Minimal code and input datasets needed to reproduce the results in the
*Bioinformatics* paper **"Benchmarking active-learning surrogates for
cell-media titer optimisation"**.

---

## What's in here

| Path | What it is |
|---|---|
| [Data/](Data/) | Six published cell-culture / cell-media titer datasets, re-formatted to a uniform schema. See [Data/SOURCES.md](Data/SOURCES.md). |
| [utils/](utils/) | The dataset registry, optimiser implementations, surrogate models, and sparse-GP approximations. |
| [experiments/](experiments/) | The two competition protocols (Hide-the-Label, Open-Race) under two difficulty regimes (Regular, Hard), plus visualisation/statistical-analysis scripts. |
| [main.py](main.py) | Convenience entry point and CLI dispatcher for the four experiment types. |
| `run_single_combo.py`, `run_supplement.py`, `run_topup.py`, `run_bnn.py` | Long-running orchestrators that generate the published result tables (see pipeline below). |

Post-processed JSON results and the raw multi-tournament logs are **not**
bundled — re-run the orchestrators below to regenerate them.

---

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch is only needed by the Bayesian-NN surrogate (`run_bnn.py`); the
classical GP / RF / NN surrogates depend only on scikit-learn.

All scripts use **relative paths** rooted at this folder, so just run them
from the package root.

---

## Reproducing the published results

The published numbers come from running each (surrogate × difficulty)
combination through both protocols (Hide-the-Label and Open-Race), at two
hidden fractions (0.95, 0.99), at three batch sizes (1, 10, 20), with **30
synthetic datasets × 30 competitions** per cell. There are eight combos:

| Combo | Surrogate | Difficulty | Runner |
|---|---|---|---|
| `GP_Regular`  | Gaussian Process       | Regular | `run_single_combo.py` → `run_supplement.py` → `run_topup.py` |
| `GP_Hard`     | Gaussian Process       | Hard    | `run_single_combo.py` → `run_supplement.py` → `run_topup.py` |
| `RF_Regular`  | Random Forest          | Regular | `run_single_combo.py` → `run_supplement.py` → `run_topup.py` |
| `RF_Hard`     | Random Forest          | Hard    | `run_single_combo.py` → `run_supplement.py` → `run_topup.py` |
| `NN_Regular`  | Neural Network         | Regular | `run_single_combo.py` → `run_supplement.py` → `run_topup.py` |
| `NN_Hard`     | Neural Network         | Hard    | `run_single_combo.py` → `run_supplement.py` → `run_topup.py` |
| `BNN_Regular` | Bayesian Neural Network| Regular | `run_bnn.py` (Phase 1 + Phase 2 in one command) |
| `BNN_Hard`    | Bayesian Neural Network| Hard    | `run_bnn.py` (Phase 1 + Phase 2 in one command) |

All runners use `random_state=42` and write outputs to
`New_Results_Official/<combo>/hiddenfrac{95,99}/{Hide_The_Label,Open_Race}/complete/`.

### Step 1 — initial 10 × 10 sweep with the 12-optimiser base pool (GP / RF / NN combos)

For the six classical-surrogate combos, generate the first 10 synth × 10
comp pass from scratch. This step runs the 12 base optimisers (it does
**not** run `SBO_ANN_PV` or `SMART_BO` — those are added in Step 2).

```bash
python run_single_combo.py --combo GP_Regular --n-jobs 2
python run_single_combo.py --combo GP_Hard    --n-jobs 2
python run_single_combo.py --combo RF_Regular --n-jobs 2
python run_single_combo.py --combo RF_Hard    --n-jobs 2
python run_single_combo.py --combo NN_Regular --n-jobs 2
python run_single_combo.py --combo NN_Hard    --n-jobs 2
```

Each combo runs ~72 jobs (6 datasets × 2 hidden fractions × 3 batch sizes
× 2 protocols) and is resumable — re-running picks up where it left off
via `New_Results_Official/_run_status_<combo>.json`. Output JSONs land in
`Results/Hard_Mode/{Hide_The_Label,Open_Race}/`.

### Step 2 — supplement (`SBO_ANN_PV` + `SMART_BO`) and bridge into `complete/`

`run_supplement.py` reads the 12-optimiser JSONs produced in Step 1, runs
the two supplement optimisers (`SBO_ANN_PV` and `SMART_BO`) at the same
scale, merges the new optimiser results into each tournament, and writes
the resulting 14-optimiser JSON to
`New_Results_Official/<combo>/hiddenfrac{95,99}/Hide_The_Label/complete/`
(HTL) and `New_Results_Official/<combo>/Open_Race/complete/` (OR). This is
the bridge between the staging area in `Results/` and the canonical
`complete/` layout that `run_topup.py` (and the published files in
`New_Results_Official/`) expect.

```bash
python run_supplement.py --combo GP_Regular --n-jobs 2
python run_supplement.py --combo GP_Hard    --n-jobs 2
python run_supplement.py --combo RF_Regular --n-jobs 2
python run_supplement.py --combo RF_Hard    --n-jobs 2
python run_supplement.py --combo NN_Regular --n-jobs 2
python run_supplement.py --combo NN_Hard    --n-jobs 2
```

After this step, each merged JSON contains `supplement_merged: true`,
`supplement_optimizers: ['SBO_ANN_PV', 'SMART_BO']`, and 14 optimisers in
every competition. **`run_topup.py` will fail with `No existing HTL JSON
…` if you skip this step.**

### Step 3 — top-up to 30 × 30 (GP / RF / NN combos)

Once Step 2 has populated `New_Results_Official/<combo>/.../complete/`,
top each combo up to the published `n_synth=30, n_comp=30` resolution:

```bash
python run_topup.py --combo GP_Regular --n-jobs 2
python run_topup.py --combo GP_Hard    --n-jobs 2
python run_topup.py --combo RF_Regular --n-jobs 2
python run_topup.py --combo RF_Hard    --n-jobs 2
python run_topup.py --combo NN_Regular --n-jobs 2
python run_topup.py --combo NN_Hard    --n-jobs 2
```

`run_topup.py` reads the 10×10 JSON from `complete/`, runs the missing 20
extra synth datasets and 20 extra competitions on each existing synth
using deterministic seed offsets, and writes the updated 30×30 JSON back
in place (`topup_applied: true`).

### Step 4 — Bayesian NN combos (single command each)

`run_bnn.py` is self-contained — it runs Phase 1 (10×10 from scratch with
all 14 optimisers), automatically bridges the results into `complete/`,
and then runs Phase 2 (top-up to 30×30):

```bash
python run_bnn.py --combo BNN_Regular --n-jobs 2
python run_bnn.py --combo BNN_Hard    --n-jobs 2
```

Unlike the GP/RF/NN combos, BNN combos do **not** need `run_supplement.py`
because Phase 1 already runs all 14 optimisers (including `SBO_ANN_PV`
and `SMART_BO`). PyTorch and a CUDA-capable GPU are recommended.

### Step 5 — generate figures and statistical tables

The statistical-analysis script expects all result JSONs for a given
setting in a single flat directory
(`<base>/Hard_Mode/Hide_The_Label/*.json`). The runner output is split
by combo; to create the flat layout needed by the analysis script, first
aggregate into `Result_Official/`:

```bash
# Aggregate runner output into flat Result_Official/ layout
for HF in hiddenfrac95 hiddenfrac99; do
  for PROTO in Hide_The_Label; do
    mkdir -p Result_Official/$HF/Hard_Mode/$PROTO
    cp New_Results_Official/*_Hard/$HF/$PROTO/complete/*.json \
       Result_Official/$HF/Hard_Mode/$PROTO/ 2>/dev/null
    mkdir -p Result_Official/$HF/Regular_Mode/$PROTO
    cp New_Results_Official/*_Regular/$HF/$PROTO/complete/*.json \
       Result_Official/$HF/Regular_Mode/$PROTO/ 2>/dev/null
  done
  for PROTO in Open_Race; do
    mkdir -p Result_Official/$HF/Hard_Mode/$PROTO
    cp New_Results_Official/*_Hard/Open_Race/complete/*.json \
       Result_Official/$HF/Hard_Mode/$PROTO/ 2>/dev/null
    mkdir -p Result_Official/$HF/Regular_Mode/$PROTO
    cp New_Results_Official/*_Regular/Open_Race/complete/*.json \
       Result_Official/$HF/Regular_Mode/$PROTO/ 2>/dev/null
  done
done
```

Then run the analysis scripts:

```bash
# Statistical analysis tables (paper Tables 1-4)
AL_RESULTS_DIR=Result_Official/hiddenfrac95 \
    python experiments/Visualization_and_Tables/statistical_analysis_tables.py

# Per-protocol bar plots (Hide-the-Label)
python experiments/Visualization_and_Tables/Hide_The_Label_bar_plots.py \
    --results Result_Official/hiddenfrac95/Hard_Mode/Hide_The_Label/<one_file>.json \
    --out figures/hide_label_mean_steps.png

# Open-Race best-so-far curves
AL_OPENRACE_JSON=Result_Official/hiddenfrac95/Hard_Mode/Open_Race/<one_file>.json \
    python experiments/Visualization_and_Tables/Open_Race_best_so_far_plots.py
```

### Quick smoke test

A small one-dataset run that verifies your install in a few minutes:

```bash
python smoke_test_nn.py
python test_optimizations.py     # validates parallel + lazy-refit code path
python test_vs_old_results.py    # rank-correlates new runs against legacy results
```

### Single ad-hoc experiment via the CLI

```bash
# Regular Mode, Hide-the-Label, MOBO rat-myocyte dataset, 5 competitions
python main.py regular-hide MOBO_dataset_rat_myocyte \
    --n_competitions 5 --hidden_fraction 0.95

# Hard Mode, Open-Race, TF-cell dataset, RF surrogate
python main.py hard-open df_Human_TF_Cell_Expanded \
    --surrogate_type random_forest --S 50 --R 5 --B 1
```

---

## Available datasets

The dataset registry is in [utils/datasets.py](utils/datasets.py). Names you
pass on the CLI:

- `MOBO_dataset_rat_myocyte`
- `DBO_dataset_rat_myocyte`
- `df_Human_Hela_regular_mode`
- `df_Human_Hela_timesaving_mode`
- `df_Human_T_Cell_Expanded`
- `df_Human_TF_Cell_Expanded`

See [Data/SOURCES.md](Data/SOURCES.md) for original publications and
licensing notes.

---

## Optimisers in the pool

`main.py` defines the canonical optimiser pool used in the paper:

- **Bayesian / surrogate-based:** `SBO_GP_PV`, `BO_GP_EI`, `SMART_BO`, `SBO_ANN_PV`, `SBO_POLY_PV`
- **Design of experiments:** `FULL_FACTORIAL`, `FRACTIONAL_FACTORIAL`, `PLACKETT_BURMAN`, `CENTRAL_COMPOSITE`, `BOX_BEHNKEN`, `LATIN_HYPERCUBE`
- **Evolutionary / random:** `DE_DIRECT`, `RANDOM`

Implementations live in [utils/optimizers.py](utils/optimizers.py). The
Hard-mode surrogates (with stochastic noise, multimodality, and
discontinuities) live in [utils/harder_surrogates.py](utils/harder_surrogates.py).

---

## Hardware notes

- Most of the published results were produced on a single NVIDIA A100 (40 GB).
  GPU is **only** required for the Bayesian-NN surrogate; everything else
  runs comfortably on CPU.
- Total wall-time for the full tournament across all 8 combos was on the
  order of several days at `n_jobs=2`. Use `--n-jobs` to scale up.

---

## Citation

If you use this code or the bundled results, please cite both the software
and the accompanying paper. See [CITATION.cff](CITATION.cff).

The software is released under the MIT licence
([LICENSE](LICENSE)). The bundled CSVs are derived from previously
published datasets — please also cite the original sources listed in
[Data/SOURCES.md](Data/SOURCES.md).
