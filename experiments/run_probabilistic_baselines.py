"""Five-seed ordinary and regression-kriging baselines.

The implementation uses the common training-only preprocessing and explicit
validation grids, avoiding test-set tuning and unstable native GP optimizers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

CODE = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))
from datasets import load_dataset
from experiments.run_baselines import make_spatial_splits


def matern32(left, right, lengthscale):
    scaled = np.sqrt(3.0) * torch.cdist(left, right) / lengthscale
    return (1.0 + scaled) * torch.exp(-scaled)


def fit_gp(coord_train, residual_train, coord_val, target_val, trend_val,
           coord_test, device):
    tr = torch.as_tensor(coord_train, dtype=torch.float32, device=device)
    va = torch.as_tensor(coord_val, dtype=torch.float32, device=device)
    te = torch.as_tensor(coord_test, dtype=torch.float32, device=device)
    residual = torch.as_tensor(residual_train, dtype=torch.float32, device=device)
    val_target = torch.as_tensor(target_val, dtype=torch.float32, device=device)
    val_trend = torch.as_tensor(trend_val, dtype=torch.float32, device=device)
    eye = torch.eye(len(tr), device=device)
    best = None
    def stable_cholesky(signal, diagonal_noise):
        for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            chol, info = torch.linalg.cholesky_ex(
                signal + (diagonal_noise + jitter) * eye)
            if int(info.max()) == 0:
                return chol
        raise RuntimeError("Matérn covariance remained non-positive-definite after jitter escalation")
    for lengthscale in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0,
                        3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        signal = matern32(tr, tr, lengthscale)
        kval = matern32(va, tr, lengthscale)
        for noise in (0.0001, 0.0005, 0.001, 0.002, 0.005,
                      0.01, 0.02, 0.05, 0.1, 0.2):
            chol = stable_cholesky(signal, noise)
            alpha = torch.cholesky_solve(residual[:, None], chol).squeeze(1)
            val_mean = kval @ alpha
            error = torch.mean((val_target - (val_trend + val_mean)).square())
            if best is None or error < best[0]:
                best = (error, lengthscale, noise, chol, alpha, val_mean)
    _, lengthscale, noise, chol, alpha, val_mean = best
    ktest = matern32(te, tr, lengthscale)
    test_mean = ktest @ alpha
    solved_val = torch.cholesky_solve(matern32(tr, va, lengthscale), chol)
    solved_test = torch.cholesky_solve(ktest.T, chol)
    val_var = (1.0 + noise - (matern32(va, tr, lengthscale) * solved_val.T).sum(1)).clamp_min(1e-8)
    test_var = (1.0 + noise - (ktest * solved_test.T).sum(1)).clamp_min(1e-8)
    return (val_mean.cpu().numpy(), val_var.cpu().numpy(),
            test_mean.cpu().numpy(), test_var.cpu().numpy(),
            lengthscale, noise)


def local_gp_predict(coord_train, residual_train, coord_query, lengthscale,
                     noise, device, neighbors=128, batch_size=128):
    """Predict using independently conditioned local Matern-3/2 GPs."""
    k = min(int(neighbors), len(coord_train))
    tree = cKDTree(coord_train)
    _, index = tree.query(coord_query, k=k)
    if k == 1:
        index = index[:, None]
    tr_coord = torch.as_tensor(coord_train, dtype=torch.float32, device=device)
    tr_residual = torch.as_tensor(residual_train, dtype=torch.float32, device=device)
    q_coord = torch.as_tensor(coord_query, dtype=torch.float32, device=device)
    all_mean, all_var = [], []
    for start in range(0, len(coord_query), batch_size):
        stop = min(start + batch_size, len(coord_query))
        ind = torch.as_tensor(index[start:stop], dtype=torch.long, device=device)
        near_coord = tr_coord[ind]
        near_residual = tr_residual[ind]
        kernel = matern32(near_coord, near_coord, lengthscale)
        eye = torch.eye(k, dtype=torch.float32, device=device).expand(len(ind), -1, -1)
        chol = None
        for jitter in (1e-5, 1e-4, 1e-3, 1e-2):
            candidate, info = torch.linalg.cholesky_ex(kernel + (noise + jitter) * eye)
            if int(info.max()) == 0:
                chol = candidate
                break
        if chol is None:
            raise RuntimeError("Local Matern covariance remained non-positive-definite after jitter escalation")
        cross = matern32(q_coord[start:stop, None, :], near_coord, lengthscale).squeeze(1)
        alpha = torch.cholesky_solve(near_residual[..., None], chol).squeeze(-1)
        mean = (cross * alpha).sum(-1)
        solved = torch.cholesky_solve(cross[..., None], chol).squeeze(-1)
        variance = (1.0 + noise - (cross * solved).sum(-1)).clamp_min(1e-8)
        all_mean.append(mean.cpu())
        all_var.append(variance.cpu())
    return torch.cat(all_mean).numpy(), torch.cat(all_var).numpy()


def fit_local_gp(coord_train, residual_train, coord_val, target_val, trend_val,
                 coord_test, device, neighbors=128):
    """Tune a local GP on validation MSE and predict validation/test points."""
    best = None
    for lengthscale in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        for noise in (0.0005, 0.002, 0.01, 0.05, 0.2):
            val_mean, val_var = local_gp_predict(
                coord_train, residual_train, coord_val, lengthscale, noise,
                device, neighbors=neighbors)
            error = np.mean((target_val - (trend_val + val_mean)) ** 2)
            if best is None or error < best[0]:
                best = (error, lengthscale, noise, val_mean, val_var)
    _, lengthscale, noise, val_mean, val_var = best
    test_mean, test_var = local_gp_predict(
        coord_train, residual_train, coord_test, lengthscale, noise, device,
        neighbors=neighbors)
    return val_mean, val_var, test_mean, test_var, lengthscale, noise


def metrics(target, prediction, variance, interval_z=1.6448536269514722):
    variance = np.maximum(variance, 1e-8)
    std = np.sqrt(variance); zscore = (target - prediction) / std
    crps = std * (zscore * (2 * ndtr(zscore) - 1)
                  + 2 * np.exp(-0.5 * zscore ** 2) / np.sqrt(2 * np.pi)
                  - 1 / np.sqrt(np.pi))
    return dict(rmse=mean_squared_error(target, prediction) ** 0.5,
                mae=mean_absolute_error(target, prediction),
                r2=r2_score(target, prediction),
                nll=np.mean(0.5 * np.log(2 * np.pi * variance)
                            + (target - prediction) ** 2 / (2 * variance)),
                crps=np.mean(crps),
                picp90=np.mean((target >= prediction-interval_z*std) & (target <= prediction+interval_z*std)),
                mpiw90=np.mean(2*interval_z*std))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "pm25", "lucas_large", "california"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1042, 1052, 1062, 1072, 1082])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional subsample size; omit to use the complete dataset")
    parser.add_argument("--inference", choices=["exact", "predictive_local", "auto"], default="auto",
                        help="Residual-GP inference; auto uses local conditioning above 5,000 training points.")
    parser.add_argument("--local-neighbors", type=int, default=128,
                        help="Number of training locations conditioned on by each local GP prediction.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--methods", nargs="+", default=["SimpleKriging", "RegressionKriging", "RFKriging"],
                        help="SimpleKriging uses the fixed zero-mean trend; OrdinaryKriging is accepted as a legacy alias")
    parser.add_argument("--data-root", type=Path, default=CODE.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=CODE.parent / "results" / "probabilistic_baselines")
    args = parser.parse_args(); rows=[]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.datasets:
        for seed in args.seeds:
            S0, X0, y0 = load_dataset(dataset, args.data_root, seed, args.max_samples)
            (S, X, y), (itr, iva, ite), scalers = make_spatial_splits(
                S0, X0, y0, seed, test_size=args.test_size, val_size=args.val_size)
            inference = ("predictive_local" if args.inference == "auto" and len(itr) > 5000
                         else "exact" if args.inference == "auto" else args.inference)
            for method in args.methods:
                if method in {"SimpleKriging", "OrdinaryKriging"}:
                    method = "SimpleKriging"
                    trend_tr=np.zeros(len(itr)); trend_va=np.zeros(len(iva)); trend_te=np.zeros(len(ite))
                elif method == "RegressionKriging":
                    design=np.c_[S, X].astype(np.float32); best=None
                    design_tr=torch.as_tensor(design[itr],device=device)
                    target_tr=torch.as_tensor(y[itr],dtype=torch.float32,device=device)
                    design_va=torch.as_tensor(design[iva],device=device)
                    target_va=torch.as_tensor(y[iva],dtype=torch.float32,device=device)
                    design_te=torch.as_tensor(design[ite],device=device)
                    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
                        augmented=torch.cat([torch.ones((len(design_tr),1),device=device),design_tr],1)
                        penalty=torch.eye(augmented.shape[1],device=device); penalty[0,0]=0
                        coefficient=torch.linalg.solve(augmented.T@augmented+alpha*penalty,
                                                       augmented.T@target_tr)
                        pred=torch.cat([torch.ones((len(design_va),1),device=device),design_va],1)@coefficient
                        err=torch.mean((target_va-pred).square())
                        if best is None or err < best[0]: best=(err, coefficient)
                    coefficient=best[1]
                    def ridge_predict(matrix):
                        return (torch.cat([torch.ones((len(matrix),1),device=device),matrix],1)@coefficient).cpu().numpy()
                    trend_tr=ridge_predict(design_tr);trend_va=ridge_predict(design_va);trend_te=ridge_predict(design_te)
                else:
                    design=np.c_[S, X]
                    reg=RandomForestRegressor(300,min_samples_leaf=2,n_jobs=-1,
                                              random_state=seed).fit(design[itr],y[itr])
                    trend_tr=reg.predict(design[itr]);trend_va=reg.predict(design[iva]);trend_te=reg.predict(design[ite])
                fitter = fit_local_gp if inference == "predictive_local" else fit_gp
                fit_kwargs = {"neighbors": args.local_neighbors} if inference == "predictive_local" else {}
                va_mean, va_var, te_mean, te_var, lengthscale, noise = fitter(
                    S[itr], y[itr]-trend_tr, S[iva], y[iva], trend_va, S[ite], device,
                    **fit_kwargs)
                va_prediction=trend_va+va_mean
                scale=np.sqrt(np.mean((y[iva]-va_prediction)**2/np.maximum(va_var,1e-8)))
                if len(iva) < 50:
                    scores=np.abs(y[iva]-va_prediction)/np.sqrt(np.maximum(va_var*max(scale,1e-4)**2,1e-8))
                    rank=min(len(scores),int(np.ceil((len(scores)+1)*0.90)))
                    interval_z=float(np.partition(scores,rank-1)[rank-1])
                else:
                    interval_z=1.6448536269514722
                prediction_std=trend_te+te_mean; variance_std=te_var*max(scale,1e-4)**2
                target_scale=float(scalers[2].scale_[0]); target_mean=float(scalers[2].mean_[0])
                prediction=prediction_std*target_scale+target_mean; target=np.asarray(y0)[ite]
                variance=variance_std*target_scale**2
                rows.append(dict(dataset=dataset,seed=seed,method=method,inference=inference,
                                 lengthscale=lengthscale,noise=noise,
                                 interval_z=interval_z,
                                 **metrics(target,prediction,variance,interval_z)))
                print(dataset, seed, method, f"RMSE={rows[-1]['rmse']:.4f}")
                args.output_dir.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output_dir/"baseline_results.csv",index=False)
                if device.type == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
