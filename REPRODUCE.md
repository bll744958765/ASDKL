# Reproducing the manuscript experiments

Run from the repository root. Generated data and figures go under `results/`, which is ignored by Git. Install dependencies and the three external datasets first. Evaluation seeds are 1042, 1052, 1062, 1072, and 1082.

| Manuscript item | Run | Generated output |
|---|---|---|
| Main point-prediction tables | `methods/ASDKL/experiments/run_main.py`; `experiments/run_baselines.py`; `experiments/summarize_main_tables.py` | ASDKL per-seed JSON, matched baseline CSV, and five-seed summaries |
| Main probabilistic table | `experiments/run_probabilistic_baselines.py`, ASDKL main runs, then `experiments/summarize_main_tables.py` | NLL, CRPS, PICP90 and MPIW90 summaries |
| Synthetic-field construction figure | `experiments/make_spatial_figures_epoch300.py --figure synthetic` | Synthetic source CSV and figure |
| LUCAS and California spatial figures | ASDKL main runs, then `experiments/make_spatial_figures_epoch300.py --figure lucas --data-root data` and `--figure california` | Prediction NPZ to figure and source CSV |
| EPA sparse learning curve and 40% point-baseline table | `experiments/run_epa_sparse.py --data-root data --output-dir results/epa_sparse_fraction_fixedtest --baseline-fraction 40 --device cuda`; then `experiments/summarize_epa_sparse_fraction.py` | Grouped-location splits, per-seed results, table and curve |
| EPA sparse probabilistic table | For fractions 10–60 and five seeds, run `experiments/run_one_epa_sparse_probabilistic_baseline.py` for each of three methods using the saved splits; then `experiments/summarize_epa_sparse_probabilistic.py` | Per-method JSON, table source CSV and LaTeX |
| Structural/module ablations | `experiments/run_ablations.py --data-root data --output-dir results/revision14_shared_relation --device cuda` | Six-block reference and ten variants |
| Kernel ablation | `methods/ASDKL/experiments/ablation/run_kernels.py --data-root data --output-dir results/revision14_kernel_variants --device cuda` | Kernel-variant runs |
| Sensitivity and ablation figures/tables | `methods/ASDKL/experiments/ablation/run_sensitivity.py --data-root data --output-dir results/revision14_sensitivity --datasets synthetic --device cuda`; then `experiments/summarize_revision14.py` | Five-seed CSV summaries and vector/raster plots |
| Convergence figure | `experiments/plot_convergence_epoch300.py` after structural reference runs | Five-seed training trajectory plot |

The architecture illustration is conceptual and needs no result data. `figure/figure.py` can replot source CSVs with `--data-dir`; its `--help` lists selected panels. The output of a full run depends on hardware and software; see `README.md`.
