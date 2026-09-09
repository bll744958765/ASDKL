# ASDKL implementation

`run.py` is the main experiment entry point. The evaluated architecture is implemented by:

- `context_encoder.py`: observed-neighbour context;
- `adaptive_fourier_encoder.py`: target-conditioned frequencies;
- `asdkl.py`: shared-frequency target/neighbour encoding, relative relation encoder, spectral attention, gated aggregation, local anchor, and nonlinear mean;
- `trend_network.py`: six residual mean blocks;
- `spectral_gp.py`: representation-space RBF plus base-coordinate RBF residual covariance, full covariance prediction, and predictive-local inference;
- `loss.py` and `train_sfnp_gp.py`: Gaussian objective, spectral regularization, checkpointing, and validation-MSE selection.

Formal defaults are 64 neighbours, 64 adaptive frequencies, representation width 64, mean width 128, six residual blocks, batch size 64, learning rate 0.003, spectral weight 0.1, and at most 300 epochs. All four complete datasets use seeds 1042, 1052, 1062, 1072, and 1082.

`experiments/ablation/run_modules.py` includes independent neighbour frequencies and removal of the target-neighbour relation encoder as separate structural ablations. `experiments/ablation/run_kernels.py` evaluates all four kernel variants on all four datasets. `experiments/sparse_observations/` contains the coordinate-grouped EPA density study.
