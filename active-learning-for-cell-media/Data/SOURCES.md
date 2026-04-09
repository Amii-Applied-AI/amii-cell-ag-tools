# Bundled datasets

The CSVs in this folder are derived from previously published cell-culture
and cell-media titer datasets. They were re-formatted into a uniform
`features … Label, Label_var` schema by the cleaning pipeline that
originally lived in `utils/Cleaning_data.py` (not bundled — it contained
absolute paths to the raw upstream files).

| File | Source | Notes |
|---|---|---|
| `DBO_dataset_rat_myocyte.csv` | Cosenza et al., rat-myocyte titer screens (DBO design) | 14 features + IS_indicator + Label/Label_var |
| `MOBO_dataset_rat_myocyte.csv` | Cosenza et al., rat-myocyte titer screens (MOBO design, expanded) | 26 components, multi-fidelity |
| `df_Human_Hela_regular_mode.csv` | Rouillard et al., HeLa media optimisation, regular mode | 30 components, A450 readout |
| `df_Human_Hela_timesaving_mode.csv` | Rouillard et al., HeLa media optimisation, time-saving mode | 30 components, A450 readout |
| `df_Human_T_Cell_Expanded.csv` | Human primary T-cell expansion screen | 16 design factors, fold-change readout |
| `df_Human_TF_Cell_Expanded.csv` | Human iPSC-derived TF-cell screen | 17 design factors, fold-change readout |

If you use any of these datasets, please cite the original publications in
addition to this reproducibility package. The transformation step is
deterministic and produces files byte-identical to the bundled CSVs.

The dataset registry that maps these CSVs to feature columns and label
columns lives in [../utils/datasets.py](../utils/datasets.py).
