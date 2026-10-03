"""Nearest-neighbour Matérn GP baseline on the common ASDKL data contract.

The estimator conditions each target on its nearest training inputs under a
Matérn-3/2 covariance on the jointly standardized coordinate--covariate
representation. It is a predictive Vecchia/NNGP-style approximation: each
test prediction uses a fixed-size local conditional solve rather than a dense
global covariance.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KDTree


class NNGPMaternRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, neighbors=64, lengthscale=0.65, noise=0.10, ridge_alpha=1.0):
        self.neighbors = int(neighbors)
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.ridge_alpha = float(ridge_alpha)

    def _kernel(self, a, b):
        distance = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
        scaled = np.sqrt(3.0) * distance / max(self.lengthscale, 1e-6)
        return (1.0 + scaled) * np.exp(-scaled)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.inputs_ = X
        self.responses_ = np.asarray(y, dtype=float)
        self.tree_ = KDTree(self.inputs_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.neighbors, len(self.inputs_))
        _, indices = self.tree_.query(X, k=k)
        predictions = np.zeros(len(X), dtype=float)
        for row, neighbour_ids in enumerate(indices):
            local_inputs = self.inputs_[neighbour_ids]
            K = self._kernel(local_inputs, local_inputs)
            K.flat[:: k + 1] += self.noise ** 2 + 1e-6
            k_star = self._kernel(X[row:row + 1], local_inputs).ravel()
            try:
                factor = cho_factor(K, lower=True, check_finite=False)
                predictions[row] += k_star @ cho_solve(factor, self.responses_[neighbour_ids], check_finite=False)
            except np.linalg.LinAlgError:
                predictions[row] += float(np.mean(self.responses_[neighbour_ids]))
        return predictions
