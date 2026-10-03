[README.md](https://github.com/user-attachments/files/33000242/README.md)
# ASDKL experimental code

This repository contains runnable code for **all experiments reported in the ASDKL manuscript**: ASDKL, 13 matched point-prediction baselines, three probabilistic kriging comparisons, the EPA observation-density study, structural and kernel ablations, sensitivity and convergence analyses, and table/figure generation. The KCN, PE-GNN, 4N, and DeepKriging comparators are the **adaptations described in the manuscript**, not unmodified upstream packages. See [REPRODUCE.md](REPRODUCE.md).

**No experimental results, raw datasets, trained checkpoints, logs, or rendered manuscript figures are included. To obtain the manuscript results, run the code yourself.** Results may differ slightly because of computer hardware, CUDA/PyTorch and other software versions, numerical libraries, and stochastic execution; such variation is normal. Use the stated seeds and protocol for the closest comparison.

## Setup

Python 3.9 and `environment.yml` were used during development. Select the PyTorch build appropriate for your hardware.

```bash
conda env create -f environment.yml
conda activate asdkl
python -m unittest discover -s tests -v
```

The synthetic dataset is generated locally. Download EPA AirData 2023, LUCAS 2018 topsoil, and California housing source data yourself and place them as specified in [data/README.md](data/README.md). Third-party datasets are not redistributed.

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

Detailed commands and the experiment-to-table/figure map are in [REPRODUCE.md](REPRODUCE.md). Run outputs go under the ignored `results/` directory. Generated values should be checked against the manuscript's means and standard deviations.

Public repository: https://github.com/bll744958765/ASDKL. A separate anonymized code package is used during double-anonymous review.

## License

No software license has yet been selected by the authors. Until one is added, GitHub's default copyright rules apply.
