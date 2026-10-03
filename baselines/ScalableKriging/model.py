"""Uniform scalable kriging baselines for all four ASDKL datasets."""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KDTree
class StableRidge:
    """Closed-form ridge regression with an unpenalized intercept."""
    def __init__(self, alpha=1.0): self.alpha=float(alpha)
    def fit(self, X, y):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        design=torch.as_tensor(np.c_[np.ones(len(X)),X],dtype=torch.float64)
        target=torch.as_tensor(y,dtype=torch.float64)
        penalty=torch.eye(design.shape[1],dtype=torch.float64)*self.alpha; penalty[0,0]=0.0
        self.coef_=torch.linalg.solve(design.T@design+penalty,design.T@target).cpu().numpy()
        return self
        return self
    def predict(self, X):
        X=np.asarray(X,dtype=float)
        design=torch.as_tensor(np.c_[np.ones(len(X)),X],dtype=torch.float64)
        coef=torch.as_tensor(self.coef_,dtype=torch.float64)
        return (design@coef).cpu().numpy()


class ScalableKrigingRegressor(BaseEstimator, RegressorMixin):
    """Local Matérn-3/2 residual kriging with a fixed-size reference set.

    ``trend`` is one of ``zero``, ``ridge`` and ``rf``.  Every variant uses the
    same 64-neighbour targetwise conditional calculation, so the comparison is
    feasible and methodologically matched from 1,000 observations to full data.
    """
    def __init__(self, trend="zero", neighbors=64, lengthscale=0.65, noise=0.10,
                 ridge_alpha=1.0, seed=42):
        self.trend = trend
        self.neighbors = int(neighbors)
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)

    def _kernel(self, a, b):
        distance = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
        scaled = np.sqrt(3.0) * distance / max(self.lengthscale, 1e-6)
        return (1.0 + scaled) * np.exp(-scaled)

    def _fit_trend(self, covariates, y):
        if self.trend == "zero":
            self.trend_model_ = None
            return np.zeros_like(y, dtype=float)
        if self.trend == "ridge":
            self.trend_model_ = StableRidge(alpha=self.ridge_alpha).fit(covariates, y)
        elif self.trend == "rf":
            self.trend_model_ = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                random_state=self.seed).fit(covariates, y)
        else:
            raise ValueError(f"Unknown trend: {self.trend}")
        return self.trend_model_.predict(covariates)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.coords_, self.covariates_ = X[:, :2], X[:, 2:]
        self.residuals_ = y - self._fit_trend(self.covariates_, y)
        self.tree_ = KDTree(self.coords_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        query_coords, query_covariates = X[:, :2], X[:, 2:]
        k = min(self.neighbors, len(self.coords_))
        _, indices = self.tree_.query(query_coords, k=k)
        prediction = (np.zeros(len(X), dtype=float) if self.trend_model_ is None
                      else self.trend_model_.predict(query_covariates))
        for row, neighbour_ids in enumerate(indices):
            local_coords = self.coords_[neighbour_ids]
            covariance = self._kernel(local_coords, local_coords)
            covariance.flat[:: k + 1] += self.noise ** 2 + 1e-6
            cross_covariance = self._kernel(query_coords[row:row + 1], local_coords).ravel()
            try:
                factor = cho_factor(covariance, lower=True, check_finite=False)
                prediction[row] += cross_covariance @ cho_solve(
                    factor, self.residuals_[neighbour_ids], check_finite=False)
            except np.linalg.LinAlgError:
                prediction[row] += float(np.mean(self.residuals_[neighbour_ids]))
        return prediction
