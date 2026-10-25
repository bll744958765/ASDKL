# ASDKL experimental code

This repository contains runnable code for **all experiments reported in the ASDKL manuscript**: ASDKL, 13 matched point-prediction baselines, three probabilistic kriging comparisons, the EPA observation-density study, structural and kernel ablations, sensitivity and convergence analyses, and table/figure generation. The KCN, PE-GNN, 4N, and DeepKriging comparators are the **adaptations described in the manuscript**, not unmodified upstream packages. See [REPRODUCE.md](REPRODUCE.md) for exact Table 1–6 and Figure 2–6/C1–C2 instructions.

**The current repository contains no separate numeric experimental results, trained checkpoints, logs, or generated manuscript plot files. To obtain the manuscript results, run the code yourself.** Three original input-data files are retained under `data/`; see [data/README.md](data/README.md) for the paths required by the loaders. The retained `docs/synthetic_example.png` is a standalone synthetic-data example, not a manuscript experiment result; regenerate it with `python examples/generate_synthetic_example.py`. Figure 1 is an architecture schematic and is not included in the plotting notebook. The notebook stores embedded previews for Figures 2–6 and C1–C2 so that their layouts can be inspected; it writes PDF/SVG/PNG files when run. Results may differ slightly because of computer hardware, CUDA/PyTorch and other software versions, numerical libraries, and stochastic execution; such variation is normal. Use the stated seeds and protocol for the closest comparison.

## Which ASDKL Python file should I run?

Run commands from the **repository root** in a terminal; do not double-click a `.py` file. For the manuscript's main ASDKL results (Tables 1–3), run `methods/ASDKL/experiments/run_main.py`. This script runs the model on the four datasets with the five stated seeds and calls `methods/ASDKL/run.py` for each individual run. For a single dataset/seed or a quick check, run `methods/ASDKL/run.py` directly. `methods/ASDKL/asdkl.py` defines the model; it is **not** an experiment entry point.

The `methods/ASDKL/experiments/sparse_observations/` folder is for the **EPA reduced-observation study**, not the main benchmark. To reproduce manuscript Figure 5 and Table 4, follow the complete command chain below starting with the repository-level `experiments/run_epa_sparse.py`; the similarly named script inside `methods/ASDKL/experiments/sparse_observations/` is an ASDKL-only runner, and `summarize_epa_sparse_fraction.py` summarizes its existing outputs. Neither replaces the full Figure 5/Table 4 workflow.

| Manuscript experiment | Run these files from the repository root, in order | Final result |
|---|---|---|
| Tables 1–2: point prediction on four datasets | `methods/ASDKL/experiments/run_main.py` (ASDKL); `experiments/run_baselines.py` (13 comparators); `experiments/summarize_main_tables.py` | `results/main_table_summaries/point_prediction_summary.csv` |
| Table 3: probabilistic prediction | Reuse the ASDKL runs above; run `experiments/run_probabilistic_baselines.py`; then `experiments/summarize_main_tables.py` | `results/main_table_summaries/probabilistic_prediction_summary.csv` |
| Figure 2: synthetic field | `experiments/prepare_spatial_figure_sources.py --figure synthetic`; then the Figure 2 cell of `figure/all_manuscript_figures.ipynb` after its data sources are available | Figure 2 plot (no model training required) |
| Figures 3–4: spatial predictions | `experiments/run_ablations.py --configurations six_block_reference`; then `experiments/prepare_spatial_figure_sources.py --figure all`; then the figure notebook | LUCAS and California plots |
| Figure 5: EPA observation density | `experiments/run_epa_sparse.py`; then the figure notebook | EPA learning-curve plot |
| Table 4: EPA probabilistic comparison | `experiments/run_epa_sparse.py`; `scripts/run_epa_probabilistic_grid.py`; `experiments/summarize_epa_sparse_probabilistic.py` | `results/epa_sparse_fraction_fixedtest/probabilistic_comparison_10_60_summary.csv` |
| Table 5 and Figure 6: structural ablations | `experiments/run_ablations.py`; `experiments/summarize_ablation_and_sensitivity.py`; then the figure notebook for Figure 6 | Structural summary and paired-ablation plot |
| Table 6: residual covariance kernels | `methods/ASDKL/experiments/ablation/run_kernels.py`; `experiments/summarize_ablation_and_sensitivity.py` | `results/asdkl_ablation_summaries/kernel_summary.csv` |
| Appendix Figure C1: convergence | `experiments/run_ablations.py --configurations six_block_reference`; then the figure notebook | Convergence plot from the training histories |
| Appendix Figure C2: sensitivity | `methods/ASDKL/experiments/ablation/run_sensitivity.py`; `experiments/summarize_ablation_and_sensitivity.py`; then the figure notebook | Sensitivity plot |

The table identifies **files**, not complete command lines: use the commands and shared output directories in [REPRODUCE.md](REPRODUCE.md) or below. Figure 1 is the architecture drawing and needs no experiment script. The notebook draws figures from newly generated local results; it does not train the models. `__init__.py` files are package markers, and the other `.py` files under `methods/ASDKL/` implement components called by the runners.

## Setup

Python 3.9 and `environment.yml` were used during development. Select the PyTorch build appropriate for your hardware.

```bash
conda env create -f environment.yml
conda activate asdkl
python -m unittest discover -s tests -v
```

The synthetic dataset is generated locally. Three input-data files are included, but they are at flat paths rather than the paths expected by the loaders. Place the provider data as specified in [data/README.md](data/README.md) before running the full experiments.

Quick CPU check:

```bash
python methods/ASDKL/run.py --dataset synthetic --seed 1042 --max-samples 120 --epochs 2 --device cpu --gp-refine-steps 0 --data-root data --output-dir results/smoke
python experiments/run_baselines.py --datasets synthetic --seeds 1042 --max-samples 120 --include SimpleKriging --data-root data --output-dir results/smoke_baseline
```

Main experiments:

```bash
python methods/ASDKL/experiments/run_main.py --data-root data --output-dir results/asdkl_main --device cuda
python experiments/run_baselines.py --data-root data --output-dir results/benchmarks
python experiments/run_probabilistic_baselines.py --data-root data --output-dir results/probabilistic_baselines
```

The default evaluation seeds are 1042, 1052, 1062, 1072, and 1082. Complete datasets use 70% training, 10% validation, and 20% test observations. ASDKL trains for at most 300 epochs and selects its checkpoint using validation MSE. Use `--device cpu` where CUDA is unavailable; complete runs can be expensive.

**Protocol distinction:** point-prediction kriging uses fixed settings (64 neighbours, Matérn-3/2 length scale 0.65, nugget 0.10). The separate probabilistic experiment selects length scale and noise using validation data and calibrates predictive variance on validation residuals. Do not mix the two result sets.

Detailed commands, original-data paths, the complete experiment-to-table/figure map, and metric definitions are in [REPRODUCE.md](REPRODUCE.md). All coded figure panels are in [figure/all_manuscript_figures.ipynb](figure/all_manuscript_figures.ipynb). Run outputs go under the ignored `results/` directory. The checked-in notebook includes seven embedded figure previews; run the experiments and execute its cells locally to regenerate and export the plots. Generated values should be checked against the manuscript's means and standard deviations.

## Reproduce each manuscript item

Run the commands below from the repository root, in order, after preparing the original datasets in [data/README.md](data/README.md). The synthetic generator needs no external file. The five default seeds are `1042 1052 1062 1072 1082`. Complete runs can take substantial GPU time. The table summary scripts check that the expected seed runs exist; they do not use numbers copied from the manuscript.

```bash
# Tables 1–3: ASDKL, 13 point baselines, and three probabilistic baselines
python methods/ASDKL/experiments/run_main.py --data-root data --output-dir results/asdkl_main --device cuda
python experiments/run_baselines.py --data-root data --output-dir results/benchmarks
python experiments/run_probabilistic_baselines.py --data-root data --output-dir results/probabilistic_baselines
python experiments/summarize_main_tables.py

# Figures 2–4: synthetic field and held-out LUCAS/California predictions
python experiments/run_ablations.py --configurations six_block_reference --data-root data --output-dir results/asdkl_module_ablations --device cuda
python experiments/prepare_spatial_figure_sources.py --figure all --data-root data

# Table 4 and Figure 5: EPA reduced-observation experiment
python experiments/run_epa_sparse.py --data-root data --output-dir results/epa_sparse_fraction_fixedtest --device cuda --baseline-fraction 40
python scripts/run_epa_probabilistic_grid.py --data-root data --results-root results/epa_sparse_fraction_fixedtest
python experiments/summarize_epa_sparse_probabilistic.py

# Tables 5–6, Figure 6, and Appendix Figures C1–C2: ablations and sensitivity
python experiments/run_ablations.py --data-root data --output-dir results/asdkl_module_ablations --device cuda
python methods/ASDKL/experiments/ablation/run_kernels.py --data-root data --output-dir results/asdkl_kernel_ablations --device cuda
python methods/ASDKL/experiments/ablation/run_sensitivity.py --data-root data --output-dir results/asdkl_sensitivity --datasets synthetic --device cuda
python experiments/summarize_ablation_and_sensitivity.py

# Generate experimental Figures 2–6 and Appendix Figures C1–C2
jupyter nbconvert --to notebook --execute figure/all_manuscript_figures.ipynb --output all_manuscript_figures_executed.ipynb --output-dir results
```

| Manuscript item | Generated source and final step |
|---|---|
| Tables 1–2 | `results/main_table_summaries/point_prediction_summary.csv`; select synthetic/EPA for Table 1 and LUCAS/California for Table 2. |
| Table 3 | `results/main_table_summaries/probabilistic_prediction_summary.csv`. |
| Table 4 | `results/epa_sparse_fraction_fixedtest/probabilistic_comparison_10_60_summary.csv`. |
| Table 5 | `results/asdkl_ablation_summaries/structural_ablation_summary.csv`. |
| Table 6 | `results/asdkl_ablation_summaries/kernel_summary.csv`. |
| Figure 1 | Architecture schematic; no experiment output or notebook cell is needed. |
| Figures 2–4 | `results/spatial_figure_source_data/` from the synthetic/LUCAS/California source-data script, then the notebook. |
| Figure 5 | `results/epa_sparse_fraction_fixedtest/asdkl_fraction_results.csv`, then the notebook. |
| Figure 6 | `results/asdkl_ablation_summaries/figure6_source.csv`, then the notebook. |
| Appendix Figure C1 | Five `synthetic_run.json` training histories under `results/asdkl_module_ablations/six_block_reference/synthetic/`, then the notebook. |
| Appendix Figure C2 | `results/asdkl_ablation_summaries/sensitivity_summary.csv`, then the notebook. |

To **reproduce** Figures 2–6 and C1–C2 after cloning, run the experiments above and execute all notebook cells. The notebook displays the figures during execution and saves the exported files under `results/`. Keep `results/` local; do not upload its CSVs, run JSONs, plots, checkpoints, or logs. The notebook's embedded image previews are the only displayed manuscript figure outputs included in the repository. See [REPRODUCE.md](REPRODUCE.md) for the exact dataset fields, splits, metrics, model settings and expected output names.

Public repository: https://github.com/bll744958765/ASDKL. A separate anonymized code package is used during double-anonymous review.

## License

No software license has yet been selected by the authors. Until one is added, GitHub's default copyright rules apply.
