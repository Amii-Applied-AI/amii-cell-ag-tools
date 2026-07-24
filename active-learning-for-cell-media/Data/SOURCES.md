# Bundled datasets

The CSVs in this folder are derived from previously published cell-culture and
cell-media titer datasets. They were reformatted into a uniform
`features ... Label, Label_var` schema; the transformation is deterministic and
reproduces the bundled CSVs exactly.

**These datasets were produced by the authors of the studies below, not by this
project.** If you use any of them, cite the original publication in addition to
this package.

| File | Original study | Notes |
|---|---|---|
| `DBO_dataset_rat_myocyte.csv` | Cosenza Z, Block DE, Baar K, Chen X. *Multi-objective Bayesian algorithm automatically discovers low-cost high-growth serum-free media for cellular agriculture application.* Engineering in Life Sciences 2023;23(8):e2300005. doi:[10.1002/elsc.202300005](https://doi.org/10.1002/elsc.202300005) | Rat myoblast titer screen; 14 features + IS indicator |
| `MOBO_dataset_rat_myocyte.csv` | Cosenza Z, Block DE, Baar K, Chen X. (as above) doi:[10.1002/elsc.202300005](https://doi.org/10.1002/elsc.202300005) | Rat myoblast screen; 26 components, multi-fidelity |
| `df_Human_Hela_regular_mode.csv` | Hashizume T, Ozawa Y, Ying B-W. *Employing active learning in the optimization of culture medium for mammalian cells.* npj Systems Biology and Applications 2023;9(20):1-12. doi:[10.1038/s41540-023-00284-7](https://doi.org/10.1038/s41540-023-00284-7) | HeLa-S3, regular mode; A450 readout |
| `df_Human_Hela_timesaving_mode.csv` | Hashizume T, Ozawa Y, Ying B-W. (as above) doi:[10.1038/s41540-023-00284-7](https://doi.org/10.1038/s41540-023-00284-7) | HeLa-S3, time-saving mode; A450 readout |
| `df_Human_T_Cell_Expanded.csv` | Kim MM, Audet J. *On-demand serum-free media formulations for human hematopoietic cell expansion using a high dimensional search algorithm.* Communications Biology 2019;2(48):1-11. doi:[10.1038/s42003-019-0296-7](https://doi.org/10.1038/s42003-019-0296-7) | Human primary T-cell expansion; fold-change readout |
| `df_Human_TF_Cell_Expanded.csv` | Kim MM, Audet J. (as above) doi:[10.1038/s42003-019-0296-7](https://doi.org/10.1038/s42003-019-0296-7) | TF-1 myeloid progenitor line; fold-change readout |

## Licensing

The CC BY 4.0 license in this repository covers **the code in this package only**. It
does not extend to the bundled datasets, which remain subject to the terms of
their original publications. All three source articles are open access and permit
reuse with attribution; consult the linked DOIs for the exact terms of each.

The dataset registry mapping these CSVs to feature and label columns is in
[../utils/datasets.py](../utils/datasets.py).
