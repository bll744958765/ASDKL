"""FourN baseline under the common ASDKL benchmark data contract."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree


class FourNRegressor:
    """Nearest-neighbour neural predictor using only training reference labels."""

    def __init__(self, seed=42, k=10):
        self.seed, self.k = int(seed), int(k)

    def _features(self, z, exclude_self=False, query_reference_indices=None):
        z = np.asarray(z)
        q = min(self.k + int(exclude_self), len(self.reference_))
        dist, idx = self.tree_.query(z[:, :2], k=q)
        if exclude_self:
            if query_reference_indices is None:
                raise ValueError("query_reference_indices is required when exclude_self=True")
            self_idx = np.asarray(query_reference_indices, dtype=int).reshape(-1)
            kept_dist = np.empty((len(idx), min(self.k, max(len(self.reference_) - 1, 0))), dtype=dist.dtype)
            kept_idx = np.empty_like(kept_dist, dtype=int)
            for row, reference_index in enumerate(self_idx):
                mask = idx[row] != reference_index
                kept_dist[row] = dist[row, mask][:self.k]
                kept_idx[row] = idx[row, mask][:self.k]
            dist, idx = kept_dist, kept_idx
        rel = self.reference_[idx, :2] - z[:, None, :2]
        cov_delta = self.reference_[idx, 2:] - z[:, None, 2:]
        local = np.concatenate(
            [rel, dist[..., None], self.response_[idx, None],
             self.reference_[idx, 2:], cov_delta], axis=2)
        return np.concatenate([z, local.reshape(len(z), -1)], axis=1)

    def fit(self, z, y):
        # Import lazily to avoid a circular import while baseline_models loads us.
        from baselines.baseline_models import TorchMLPRegressor
        self.reference_, self.response_ = np.asarray(z), np.asarray(y)
        self.tree_ = KDTree(self.reference_[:, :2])
        self.network_ = TorchMLPRegressor((128, 64), seed=self.seed)
        self.network_.fit(
            self._features(self.reference_, exclude_self=True,
                           query_reference_indices=np.arange(len(self.reference_))),
            self.response_)
        return self

    def predict(self, z):
        return self.network_.predict(self._features(z))
