# Reproducing the manuscript experiments (Figures 2–6, C1–C2; Tables 1–6)

Run every command from the repository root. The canonical figure workflow is the single notebook `figure/all_manuscript_figures.ipynb`. Generated files go to the ignored `results/` directory. Follow the numbered items below for the current manuscript rather than treating a plotting CSV as a complete reproduction.

## Exact workflow for the current IJGIS manuscript

The current manuscript has **six numbered tables, six numbered figures, and two appendix figures (C1 and C2)**. The older generic item ranges in the editor's letter are examples, not the current numbering. All commands below are run from the repository root. `results/` is generated locally and is excluded from the upload package. Use a single Python environment and keep the five default seeds: `1042 1052 1062 1072 1082`. A GPU is recommended for the 300-epoch model runs; replace `cuda` with `cpu` if needed. The CPU choice changes speed, not the named workflow.

**Entry-point rule:** `methods/ASDKL/experiments/run_main.py` launches the main ASDKL benchmark (Tables 1–3) and calls `methods/ASDKL/run.py` for each dataset and seed. `methods/ASDKL/asdkl.py` defines the model and is not run directly. The EPA sparse-observation manuscript workflow begins with the repository-root `experiments/run_epa_sparse.py`; the similarly named runner under `methods/ASDKL/experiments/sparse_observations/` is ASDKL-only and does not produce the full Table 4 comparison. See the experiment-to-file table in [README.md](README.md), then use the exact command chains below.

### 0. Source data and validation

1. Create the environment: `conda env create -f environment.yml`, then `conda activate asdkl`. Alternatively install `requirements.txt` and `notebook`, `nbclient`, `ipykernel` into Python 3.9+.
2. Obtain EPA and LUCAS from their providers as specified at the **relative paths** in [data/README.md](data/README.md), then run `python scripts/prepare_california.py` for California housing. The synthetic dataset is generated in `common/synthetic_dataset.py` from each seed.
3. Check loading and syntax with `python -m unittest discover -s tests -v`. The loader in `common/datasets.py` selects the stated EPA monitor records, LUCAS complete cases, and California fields. `common/data_loader.py` creates fixed 70/10/20 splits and fits all imputation/scalers on training records only. Training observations, never validation/test targets, form neighbourhood features.
4. Do not use a plotting CSV as a substitute for the original dataset. The commands below create all training outputs and then the notebook creates the figures. No manual changes to paths or code are needed.

### Tables 1–3: complete-data benchmark and probabilistic comparison

Run this shared chain once:

```bash
python methods/ASDKL/experiments/run_main.py --data-root data --output-dir results/asdkl_main --device cuda
python experiments/run_baselines.py --data-root data --output-dir results/benchmarks
python experiments/run_probabilistic_baselines.py --data-root data --output-dir results/probabilistic_baselines
python experiments/summarize_main_tables.py
```

`run_main.py` trains ASDKL for up to 300 epochs on four datasets and five seeds, retaining the validation-MSE-selected checkpoint. The 3,000-point synthetic and EPA runs explicitly use full residual-covariance inference and refine covariance hyperparameters on all training residuals after checkpoint selection; LUCAS and California use predictive-local inference. The synthetic full calculation uses 2,100 training observations and can require more GPU memory and time than the local mode. `run_baselines.py` fits the **13 point-prediction comparators** on identical seed/split settings. `run_probabilistic_baselines.py` fits SimpleKriging, RegressionKriging, and RFKriging with validation-based covariance/variance settings. `summarize_main_tables.py` fails if a dataset/method lacks a seed. It writes `results/main_table_summaries/point_prediction_source.csv`, `point_prediction_summary.csv`, `probabilistic_prediction_source.csv`, and `probabilistic_prediction_summary.csv`.

| Manuscript item | Rows and metrics to use from the generated summary |
|---|---|
| Table 1 | `point_prediction_summary.csv`: datasets `synthetic` and `pm25`; RMSE, MAE, R² means and sample SDs across five seeds. |
| Table 2 | Same file: `lucas_large` and `california`; same metrics. |
| Table 3 | `probabilistic_prediction_summary.csv`: all four datasets; NLL, CRPS, PICP90 and MPIW90 means and sample SDs. The ASDKL metrics come from the same five main runs. |

### Figures 2–4: synthetic and held-out spatial panels

Figure 2 (synthetic generator) does not require model training. To generate its source CSV from seed 1042, run `python experiments/prepare_spatial_figure_sources.py --figure synthetic --data-root data`. Figures 3–4 use held-out test predictions from the **six-block reference** at seed 1042. Train that reference before generating their spatial source data:

```bash
python experiments/run_ablations.py --configurations six_block_reference --data-root data --output-dir results/asdkl_module_ablations --device cuda
python experiments/prepare_spatial_figure_sources.py --figure all --data-root data
```

The second command creates `results/spatial_figure_source_data/` with the synthetic field, LUCAS held-out observations/ASDKL/HistGBR predictions, and California held-out observations/ASDKL predictions. The plotting notebook reads these files and exports `synthetic_field_visualization_epoch300`, `lucas_full_asdkl_histgbr_comparison_epoch300`, and `california_full_spatial_visualization_epoch300`. Figure 2 displays a 1,000-location synthetic realization; panel (c) uses the 1st–99th percentile colour range stated in the caption. Spatial colour limits are only for display; all Table 1–3 metrics use original, unclipped predictions and responses.

### Figure 5 and Table 4: EPA reduced observations

```bash
python experiments/run_epa_sparse.py --data-root data --output-dir results/epa_sparse_fraction_fixedtest --device cuda --baseline-fraction 40
python scripts/run_epa_probabilistic_grid.py --data-root data --results-root results/epa_sparse_fraction_fixedtest
python experiments/summarize_epa_sparse_probabilistic.py
```

The first command fixes grouped-location validation/test partitions per seed and trains nested 10–70% observation subsets. It writes `asdkl_fraction_results.csv` plus `fraction_*/splits/seed_*.npz`. The grid script runs three probabilistic kriging baselines for 10, 20, 30, 40, 50 and 60% against those **same frozen splits** (90 cells); it stops on missing splits or a failed run. The summarizer checks 5 seeds per method/fraction and writes `results/epa_sparse_fraction_fixedtest/probabilistic_comparison_10_60_summary.csv`, which is **Table 4** (NLL, CRPS, PICP90, MPIW90). The notebook reads `asdkl_fraction_results.csv` directly for **Figure 5**: five seed curves, the mean RMSE and its 95% t interval, and recovered improvement relative to 10% and 70%.

### Figure 6, Tables 5–6, and Appendix Figures C1–C2

Complete the structural/module, kernel, and sensitivity runs (the six-block reference already created above can be reused):

```bash
python experiments/run_ablations.py --data-root data --output-dir results/asdkl_module_ablations --device cuda
python methods/ASDKL/experiments/ablation/run_kernels.py --data-root data --output-dir results/asdkl_kernel_ablations --device cuda
python methods/ASDKL/experiments/ablation/run_sensitivity.py --data-root data --output-dir results/asdkl_sensitivity --datasets synthetic --device cuda
python experiments/summarize_ablation_and_sensitivity.py
```

`summarize_ablation_and_sensitivity.py` checks the expected per-seed run JSONs and writes `results/asdkl_ablation_summaries/`. **Table 5** uses `structural_ablation_summary.csv` (four datasets, five seeds, test RMSE). **Table 6** uses `kernel_summary.csv` (four datasets; RMSE, NLL, CRPS). **Figure 6** uses `figure6_source.csv`, paired by dataset and seed to the complete model. **Appendix C2** uses `sensitivity_summary.csv`, where the plotted ratio is validation RMSE divided by the same-seed default. **Appendix C1** reads `results/asdkl_module_ablations/six_block_reference/synthetic/seed_*/synthetic_run.json` training histories and selected epochs directly. No test target is used in sensitivity-based selection.

For the corrected nonstationary synthetic analysis, run the Table 1–3 chain and the Figure 6/Tables 5–6/C1–C2 chain above from a clean checkout without older generated `results/` files. The synthetic runners explicitly request `--gp-inference exact`; they refine covariance parameters for 30 steps using all 2,100 training residuals after selecting the neural checkpoint. Each generated run JSON records `gp_inference` and `gp_refine_steps_applied`. The two table summarizers reject stale predictive-local synthetic records. The `no_gp` ablation correctly records `gp_inference=none` and zero refinement steps. All records, summaries, checkpoints, and generated figures remain local under the ignored `results/` directory.

### Generate every experimental figure, including vector files

Open `figure/all_manuscript_figures.ipynb` with repository root as the working directory and run **all cells in order**. For command-line execution without changing the notebook:

```bash
jupyter nbconvert --to notebook --execute figure/all_manuscript_figures.ipynb --output all_manuscript_figures_executed.ipynb --output-dir results
```

Figure 1 is an architecture schematic; its artwork and notebook cell are intentionally excluded from this code update. The notebook cells for Figures 2–6 and C1–C2 assert that their source files are present. PDF, SVG and PNG are written to `results/generated_figures/`; this path is excluded from Git. The checked-in notebook includes display outputs for inspection, while numeric experiment result files, checkpoints and generated figure files remain excluded from this update. The notebook uses 13 pt source labels in main experimental figures and 14 pt in the narrower appendix figures. Audit the compiled manuscript PDF after any figure replacement because the editor requires **at least 9 pt as displayed in the document**, not merely in the source PDF.

### Metric definitions and expected variation

RMSE, MAE and R² are point-prediction assessments on held-out test records. NLL and CRPS assess the calibrated predictive Gaussian distribution; PICP90 is empirical coverage of nominal 90% intervals and MPIW90 their mean width. Scripts save per-seed values before computing arithmetic means and **sample** standard deviations (`ddof=1`). Compare the generated source and summary files to the manuscript tables; do not copy rounded manuscript numbers into code. Different hardware, CUDA/PyTorch, numerical libraries and operating systems can change stochastic training results slightly. Such variation is normal, but missing rows, failed statuses, or changed split/seeding should be treated as reproduction failures.
