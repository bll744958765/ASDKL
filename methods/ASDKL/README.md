# ASDKL implementation

To reproduce the manuscript's **main ASDKL results**, run `python methods/ASDKL/experiments/run_main.py --data-root data --output-dir results/asdkl_main --device cuda` from the repository root. This wrapper calls `run.py` once per dataset and seed. Run `methods/ASDKL/run.py` directly only for an individual run or a quick check. `asdkl.py` contains the model definition and is not a script to launch by double-clicking. See the repository-root `README.md` and `REPRODUCE.md` for the full table/figure workflows.

| ASDKL experiment | File to run | Manuscript item |
|---|---|---|
| Four-dataset, five-seed main benchmark | `methods/ASDKL/experiments/run_main.py` | ASDKL rows in Tables 1–3 |
| Six-block reference and structural changes | `experiments/run_ablations.py` at repository root | Figures 3–4 (reference predictions), Table 5, Figure 6, Appendix C1 |
| EPA reduced-observation series | `experiments/run_epa_sparse.py` at repository root | Figure 5 and ASDKL inputs for Table 4 |
| Residual covariance variants | `methods/ASDKL/experiments/ablation/run_kernels.py` | Table 6 |
| Hyperparameter sensitivity | `methods/ASDKL/experiments/ablation/run_sensitivity.py` | Appendix C2 |

The `experiments/sparse_observations/run_epa_sparse_fraction_study.py` file is an ASDKL-only alternative for the EPA fraction series. The manuscript's complete Figure 5/Table 4 workflow uses the repository-root `experiments/run_epa_sparse.py` and its follow-up baseline and summary scripts. `experiments/sparse_observations/summarize_epa_sparse_fraction.py` reads results made by the ASDKL-only alternative; it is not the main training entry point.

The evaluated architecture is implemented by:

- `context_encoder.py`: observed-neighbour context;
- `adaptive_fourier_encoder.py`: target-conditioned frequencies;
- `asdkl.py`: shared-frequency target/neighbour encoding, relative relation encoder, spectral attention, gated aggregation, local anchor, and nonlinear mean;
- `trend_network.py`: six residual mean blocks;
- `spectral_gp.py`: representation-space RBF plus base-coordinate RBF residual covariance, full covariance prediction, and predictive-local inference;
- `loss.py` and `train_asdkl_gp.py`: Gaussian objective, spectral regularization, checkpointing, and validation-MSE selection.

Formal defaults are 64 neighbours, 64 adaptive frequencies, representation width 64, mean width 128, six residual blocks, batch size 64, learning rate 0.003, spectral weight 0.1, and at most 300 epochs. All four complete datasets use seeds 1042, 1052, 1062, 1072, and 1082.

`experiments/ablation/run_modules.py` includes independent neighbour frequencies and removal of the target-neighbour relation encoder as separate structural ablations. `experiments/ablation/run_kernels.py` evaluates all four kernel variants on all four datasets. `experiments/sparse_observations/` contains an ASDKL-only coordinate-grouped EPA density runner and its summarizer. For the manuscript's complete Figure 5/Table 4 workflow, start instead with the repository-level `experiments/run_epa_sparse.py` and follow `REPRODUCE.md`.
