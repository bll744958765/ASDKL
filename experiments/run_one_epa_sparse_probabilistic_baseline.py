"""One EPA sparse-observation probabilistic baseline on a frozen grouped split."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

CODE = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(CODE))
from datasets import load_dataset
from data_loader import preprocess_spatial_split
from run_probabilistic_baselines import fit_gp, metrics

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["SimpleKriging", "RegressionKriging", "RFKriging"], required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--training-fraction-pct", type=int, required=True)
p.add_argument("--split-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--data-root", type=Path, default=CODE.parent / "data")
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
out = a.output_dir / f"{a.method}_seed_{a.seed}.json"
if out.exists():
    raise SystemExit(0)

S0, X0, y0 = load_dataset("pm25", a.data_root, a.seed, None)
split = np.load(a.split_dir / f"seed_{a.seed}.npz")
(S, X, y), (itr, iva, ite), scalers = preprocess_spatial_split(
    S0, X0, y0, split["train_idx"], split["val_idx"], split["test_idx"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
design = np.c_[S, X].astype(np.float32)

if a.method == "SimpleKriging":
    trend_tr = np.zeros(len(itr)); trend_va = np.zeros(len(iva)); trend_te = np.zeros(len(ite))
elif a.method == "RegressionKriging":
    dtr = torch.as_tensor(design[itr], device=device)
    dva = torch.as_tensor(design[iva], device=device)
    dte = torch.as_tensor(design[ite], device=device)
    ytr = torch.as_tensor(y[itr], dtype=torch.float32, device=device)
    yva = torch.as_tensor(y[iva], dtype=torch.float32, device=device)
    best = None
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
        aug = torch.cat([torch.ones((len(dtr), 1), device=device), dtr], 1)
        penalty = torch.eye(aug.shape[1], device=device); penalty[0, 0] = 0
        coef = torch.linalg.solve(aug.T @ aug + alpha * penalty, aug.T @ ytr)
        pred = torch.cat([torch.ones((len(dva), 1), device=device), dva], 1) @ coef
        err = torch.mean((yva - pred).square())
        if best is None or err < best[0]: best = (err, coef)
    coef = best[1]
    def rp(mat):
        return (torch.cat([torch.ones((len(mat), 1), device=device), mat], 1) @ coef).cpu().numpy()
    trend_tr, trend_va, trend_te = rp(dtr), rp(dva), rp(dte)
else:
    reg = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=a.seed)
    reg.fit(design[itr], y[itr])
    trend_tr, trend_va, trend_te = reg.predict(design[itr]), reg.predict(design[iva]), reg.predict(design[ite])

start = time.perf_counter()
va_mean, va_var, te_mean, te_var, lengthscale, noise = fit_gp(
    S[itr], y[itr] - trend_tr, S[iva], y[iva], trend_va, S[ite], device)
va_pred = trend_va + va_mean
scale = np.sqrt(np.mean((y[iva] - va_pred) ** 2 / np.maximum(va_var, 1e-8)))
if len(iva) < 50:
    scores = np.abs(y[iva] - va_pred) / np.sqrt(np.maximum(va_var * max(scale, 1e-4) ** 2, 1e-8))
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * 0.90)))
    interval_z = float(np.partition(scores, rank - 1)[rank - 1])
else:
    interval_z = 1.6448536269514722
pred_std = trend_te + te_mean
var_std = te_var * max(scale, 1e-4) ** 2
target_scale = float(scalers[2].scale_[0]); target_mean = float(scalers[2].mean_[0])
pred = pred_std * target_scale + target_mean
target = np.asarray(y0)[ite]
variance = var_std * target_scale ** 2
row = dict(dataset="pm25", training_fraction_pct=a.training_fraction_pct,
           seed=a.seed, method=a.method, inference="exact",
           lengthscale=float(lengthscale), noise=float(noise),
           variance_scale=float(scale), interval_z=float(interval_z),
           seconds=time.perf_counter()-start,
           **metrics(target, pred, variance, interval_z))
out.write_text(json.dumps(row, indent=2), encoding="utf-8")
print(json.dumps(row), flush=True)
