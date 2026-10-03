"""Matched spatial-prediction baselines.

Classical estimators use scikit-learn. Neural baselines are compact PyTorch
adaptations designed around the common ASDKL data contract. The adaptations and fixed experimental settings are documented in the manuscript.
"""
from __future__ import annotations
import numpy as np
import copy
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KDTree
from pathlib import Path
import importlib.util


def _load_local_baseline(folder, filename, class_name):
    path = Path(__file__).parent / folder / filename
    spec = importlib.util.spec_from_file_location(f"baseline_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


NNGPMaternRegressor = _load_local_baseline("NNGP-Matern", "model.py", "NNGPMaternRegressor")
SparseGPNystromRegressor = _load_local_baseline("SparseGP-Nystrom", "model.py", "SparseGPNystromRegressor")
ScalableKrigingRegressor = _load_local_baseline("ScalableKriging", "model.py", "ScalableKrigingRegressor")
FourNRegressor = _load_local_baseline("4N", "model.py", "FourNRegressor")


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Small deterministic PyTorch regressor with internal early stopping."""
    def __init__(self, hidden=(128, 64), seed=42, max_epochs=300,
                 patience=30, lr=1e-3):
        self.hidden, self.seed = tuple(hidden), int(seed)
        self.max_epochs, self.patience, self.lr = max_epochs, patience, lr

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.mean_, self.scale_ = X.mean(0), X.std(0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        X = (X - self.mean_) / self.scale_
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X)); n_val = max(1, int(0.1 * len(X)))
        va, tr = order[:n_val], order[n_val:]
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers=[]; width=X.shape[1]
        for hidden_width in self.hidden:
            layers.extend([torch.nn.Linear(width, hidden_width), torch.nn.GELU()])
            width=hidden_width
        layers.append(torch.nn.Linear(width, 1))
        self.model_=torch.nn.Sequential(*layers).to(device)
        xt=torch.as_tensor(X[tr],device=device); yt=torch.as_tensor(y[tr,None],device=device)
        xv=torch.as_tensor(X[va],device=device); yv=torch.as_tensor(y[va,None],device=device)
        optimizer=torch.optim.AdamW(self.model_.parameters(),lr=self.lr,weight_decay=1e-4)
        best=float("inf"); best_state=None; stale=0
        for _ in range(self.max_epochs):
            self.model_.train(); optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(self.model_(xt),yt); loss.backward(); optimizer.step()
            self.model_.eval()
            with torch.no_grad(): val=float(torch.nn.functional.mse_loss(self.model_(xv),yv))
            if val < best - 1e-7:
                best, best_state, stale = val, copy.deepcopy(self.model_.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience: break
        self.model_.load_state_dict(best_state); self.model_.eval(); self.device_=device
        return self

    def predict(self, X):
        X=(np.asarray(X,dtype=np.float32)-self.mean_)/self.scale_
        with torch.no_grad():
            return self.model_(torch.as_tensor(X,device=self.device_)).squeeze(1).cpu().numpy()


def make_rbf_features(coords, centers, scales=(0.15, 0.3, 0.6)):
    blocks=[]
    for scale in scales:
        d2=((coords[:,None,:]-centers[None,:,:])**2).sum(-1)
        blocks.append(np.exp(-d2/(2*scale**2)))
    return np.concatenate(blocks,axis=1)


class IDWRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, power=2.0, k=16): self.power,self.k=power,k
    def fit(self,X,y): self.X_,self.y_=np.asarray(X)[:,:2],np.asarray(y); return self
    def predict(self,X):
        q=np.asarray(X)[:,:2]; d=np.sqrt(((q[:,None,:]-self.X_[None,:,:])**2).sum(-1))
        idx=np.argpartition(d,min(self.k,len(self.X_)-1),axis=1)[:,:self.k]; ds=np.take_along_axis(d,idx,axis=1)
        w=1/(ds+1e-8)**self.power; return (w*np.take(self.y_,idx)).sum(1)/w.sum(1)


class SpatialFeatureRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, mode="deepkriging", seed=42, n_centers=64): self.mode,self.seed,self.n_centers=mode,seed,n_centers
    def _features(self,X):
        coord,aux=X[:,:2],X[:,2:]
        if self.mode=="fourier":
            rng=np.random.default_rng(self.seed); W=rng.normal(0,2.0,(2,64)); z=2*np.pi*coord@W
            return np.c_[aux,np.sin(z),np.cos(z)]
        if self.mode=="deepkriging": return np.c_[aux,make_rbf_features(coord,self.centers_)]
        return X
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); X=np.asarray(X)
        self.centers_=X[rng.choice(len(X),min(self.n_centers,len(X)),replace=False),:2]
        self.model_=TorchMLPRegressor((128,64),seed=self.seed)
        self.model_.fit(self._features(X),y); return self
    def predict(self,X): return self.model_.predict(self._features(np.asarray(X)))


class NeighborhoodNeuralRegressor(BaseEstimator, RegressorMixin):
    """Common-data adaptation of KCN/PE-GNN using training-only neighbours."""
    def __init__(self,mode="kcn",seed=42,k=16): self.mode,self.seed,self.k=mode,seed,k
    def _features(self,X,exclude_self=False,query_reference_indices=None):
        X=np.asarray(X); coord=X[:,:2]; q=min(self.k+int(exclude_self),len(self.ref_)); dist,idx=self.tree_.query(coord,k=q)
        if exclude_self:
            if query_reference_indices is None:
                raise ValueError("query_reference_indices is required when exclude_self=True")
            self_idx=np.asarray(query_reference_indices,dtype=int).reshape(-1)
            kept_dist=np.empty((len(idx),min(self.k,max(len(self.ref_)-1,0))),dtype=dist.dtype)
            kept_idx=np.empty_like(kept_dist,dtype=int)
            for row,reference_index in enumerate(self_idx):
                mask=idx[row]!=reference_index
                kept_dist[row]=dist[row,mask][:self.k]
                kept_idx[row]=idx[row,mask][:self.k]
            dist,idx=kept_dist,kept_idx
        vals=self.yref_[idx]; w=np.exp(-dist/(np.median(dist)+1e-6)); agg=(w*vals).sum(1)/(w.sum(1)+1e-8)
        stats=np.c_[agg,vals.mean(1),vals.std(1),dist.mean(1),dist.min(1)]
        if self.mode=="pegnn":
            pos=np.concatenate([np.sin(2*np.pi*coord*f) for f in (1,2,4,8)]+[np.cos(2*np.pi*coord*f) for f in (1,2,4,8)],axis=1)
            return np.c_[X[:,2:],pos,stats]
        return np.c_[X,stats]
    def fit(self,X,y):
        self.ref_,self.yref_=np.asarray(X)[:,:2],np.asarray(y); self.tree_=KDTree(self.ref_)
        self.model_=TorchMLPRegressor((128,64),seed=self.seed)
        self.model_.fit(self._features(X,True,np.arange(len(X))),y); return self
    def predict(self,X): return self.model_.predict(self._features(X,False))


def build_baselines(seed=42):
    return {
      "IDW": IDWRegressor(),
      "GPR-Matern": GaussianProcessRegressor(ConstantKernel(1.0)*Matern(nu=1.5)+WhiteKernel(.1),normalize_y=True,random_state=seed),
      "SimpleKriging": ScalableKrigingRegressor(trend="zero", seed=seed),
      "RegressionKriging": ScalableKrigingRegressor(trend="ridge", seed=seed),
      "RFKriging": ScalableKrigingRegressor(trend="rf", seed=seed),
      "NNGP-Matern": NNGPMaternRegressor(),
      "SparseGP-Nystrom": SparseGPNystromRegressor(seed=seed),
      "SVR-RBF": make_pipeline(StandardScaler(),SVR(C=10,epsilon=.05)),
      "RandomForest": RandomForestRegressor(300,min_samples_leaf=2,n_jobs=-1,random_state=seed),
      "ExtraTrees": ExtraTreesRegressor(300,min_samples_leaf=2,n_jobs=-1,random_state=seed),
      "HistGBR": HistGradientBoostingRegressor(max_iter=300,l2_regularization=1e-3,random_state=seed),
      "Coordinate-MLP": TorchMLPRegressor((128,64),seed=seed),
      "Fourier-MLP": SpatialFeatureRegressor("fourier",seed),
      "DeepKriging": SpatialFeatureRegressor("deepkriging",seed),
      "4N-adapted": FourNRegressor(seed=seed),
      "KCN-adapted": NeighborhoodNeuralRegressor("kcn",seed),
      "PE-GNN-adapted": NeighborhoodNeuralRegressor("pegnn",seed),
      "KernelRidge": make_pipeline(StandardScaler(),KernelRidge(alpha=.1,kernel="rbf",gamma=.1)),
    }
