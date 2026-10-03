"""Inducing-point Nyström Gaussian-process baseline for matched benchmarks."""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import MiniBatchKMeans
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



class SparseGPNystromRegressor(BaseEstimator, RegressorMixin):
    """Low-rank RBF GP using deterministic inducing centres and a ridge nugget."""
    def __init__(self, inducing_points=256, lengthscale=0.65, noise=0.10, ridge_alpha=1.0, seed=42, center_method="minibatch"):
        self.inducing_points = int(inducing_points)
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)
        self.center_method = center_method

    def _kernel(self, a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / max(self.lengthscale, 1e-6) ** 2)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.trend_ = StableRidge(alpha=self.ridge_alpha).fit(X[:, 2:], y)
        residuals = y - self.trend_.predict(X[:, 2:])
        coordinates = X[:, :2]
        m = min(self.inducing_points, len(coordinates))
        if m == len(X):
            self.inducing_ = coordinates.copy()
        else:
            if self.center_method == "minibatch":
                centres=MiniBatchKMeans(n_clusters=m,batch_size=min(2048,len(X)),n_init=3,random_state=self.seed)
                self.inducing_=centres.fit(coordinates).cluster_centers_
            elif self.center_method == "lloyd":
                rng=np.random.default_rng(self.seed)
                centres=coordinates[rng.choice(len(coordinates),m,replace=False)].copy()
                for _ in range(20):
                    d2=((coordinates[:,None,:]-centres[None,:,:])**2).sum(-1)
                    labels=np.argmin(d2,axis=1)
                    updated=centres.copy()
                    for cluster in range(m):
                        members=coordinates[labels==cluster]
                        if len(members): updated[cluster]=members.mean(axis=0)
                    if np.max(np.abs(updated-centres))<1e-6: break
                    centres=updated
                self.inducing_=centres
            else:
                raise ValueError(f"Unknown center_method: {self.center_method}")
        Knm = self._kernel(coordinates, self.inducing_)
        Kmm = self._kernel(self.inducing_, self.inducing_)
        Knm_t=torch.as_tensor(Knm,dtype=torch.float64)
        Kmm_t=torch.as_tensor(Kmm,dtype=torch.float64)
        residual_t=torch.as_tensor(residuals,dtype=torch.float64)
        system=Knm_t.T@Knm_t+(self.noise**2)*Kmm_t
        system=system+1e-6*torch.eye(m,dtype=torch.float64)
        self.weights_=torch.linalg.solve(system,Knm_t.T@residual_t).cpu().numpy()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        kernel=torch.as_tensor(self._kernel(X[:, :2],self.inducing_),dtype=torch.float64)
        weights=torch.as_tensor(self.weights_,dtype=torch.float64)
        return self.trend_.predict(X[:,2:])+(kernel@weights).cpu().numpy()
