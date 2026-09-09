"""Reproducible ASDKL experiment entry point."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPOSITORY_ROOT / "common"))
import numpy as np
import torch
from scipy.special import ndtr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from asdkl import ASDKL
from data_loader import SpatialNeighborhoodDataset, make_spatial_splits, preprocess_spatial_split
from datasets import load_dataset
from spectral_gp import SpectralGP
from train_sfnp_gp import train_sfnp_gp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "california", "lucas", "lucas_large", "pm25", "meuse"], default="synthetic")
    p.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results" / "asdkl")
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--neighbors", type=int, default=64); p.add_argument("--freq-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64); p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--fusion", choices=["neighbor", "gated"], default="gated")
    p.add_argument("--trend-width", type=int, default=128)
    p.add_argument("--trend-depth", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--trend-residual", type=int, choices=[0, 1], default=1)
    p.add_argument("--local-anchor", type=int, choices=[0, 1], default=1)
    p.add_argument("--lambda-gp", type=float, default=1.0, help="GP-NLL weight for the optional hybrid objective")
    p.add_argument("--lambda-mse", type=float, default=0.0,
                   help="Optional auxiliary MSE weight; zero is the validated default")
    p.add_argument("--lambda-spectral", type=float, default=0.1)
    p.add_argument("--objective", choices=["joint_nll", "hybrid_mse_nll"], default="joint_nll",
                   help="joint_nll is the legacy CLI name for the response-conditioned Gaussian composite objective; hybrid is retained for sensitivity analysis")
    p.add_argument("--max-samples", type=int); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Held-out test fraction; use 0.90 for the sparse-label stress test")
    p.add_argument("--val-size", type=float, default=0.1,
                   help="Validation fraction of all observations (not of the training remainder)")
    p.add_argument("--split-file", type=Path, help="Optional NPZ containing train_idx, val_idx and test_idx")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gp-inference", choices=["auto", "exact", "nngp"], default="auto")
    p.add_argument("--gp-neighbors", type=int, default=32)
    p.add_argument("--representation-kernel", type=int, choices=[0, 1], default=1)
    p.add_argument("--base-kernel", choices=["matern32", "rbf", "none"], default="rbf")
    p.add_argument("--gp-refine-steps", type=int, default=30,
                   help="Full-training covariance hyperparameter refinement for moderate datasets")
    p.add_argument("--residual-ensemble", type=int, choices=[0, 1], default=0,
                   help="Validation-weighted spectral GP and coordinate-covariate regression kriging")
    p.add_argument("--local-expert", type=int, choices=[0, 1], default=0,
                   help="Validation-gated geographic local interpolation expert")
    p.add_argument("--covariate-expert", type=int, choices=[0, 1], default=0,
                   help="Validation-gated ExtraTrees--Matérn safeguard for covariate-dominant tasks")
    p.add_argument("--restore-best", type=int, choices=[0, 1], default=0,
                   help="Restore the validation-MSE checkpoint for prediction after saving the last-epoch training state")
    p.add_argument("--resume", type=int, choices=[0, 1], default=1,
                   help="Continue from the resumable state in the output directory when available")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Overwrite the resumable training-state checkpoint every N epochs")
    p.add_argument("--evaluate-only", type=int, choices=[0, 1], default=0,
                   help="Recompute predictions from an existing selected checkpoint without training")
    p.add_argument("--ablation",choices=["full","no_context","fixed_fourier","no_fourier","independent_frequencies","no_relation_encoder","no_spectral_attention","mean_aggregation","no_gp","no_spectral_reg"],default="full")
    return p.parse_args()


def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    S, X, y = load_dataset(args.dataset, args.data_root, args.seed, args.max_samples)
    if args.split_file is None:
        (S, X, y), (itr, iva, ite), scalers = make_spatial_splits(
            S, X, y, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    else:
        split = np.load(args.split_file)
        (S, X, y), (itr, iva, ite), scalers = preprocess_spatial_split(
            S, X, y, split["train_idx"], split["val_idx"], split["test_idx"])
    def ds(idx, self_ref=False):
        reference_indices = np.arange(len(itr)) if self_ref else None
        return SpatialNeighborhoodDataset(
            S[idx], X[idx], y[idx], S[itr], X[itr], y[itr],
            args.neighbors, self_ref, reference_indices)
    train, val, test = ds(itr, True), ds(iva), ds(ite)
    loaders = {"train": DataLoader(train, args.batch_size, shuffle=True), "val": DataLoader(val, args.batch_size), "test": DataLoader(test, args.batch_size)}
    device = torch.device(args.device)
    model = ASDKL(2, X.shape[1], freq_dim=args.freq_dim,
                  hidden_dim=args.hidden_dim, ablation=args.ablation,
                  fusion=args.fusion, trend_width=args.trend_width,
                  trend_depth=args.trend_depth, dropout=args.dropout,
                  trend_residual=bool(args.trend_residual),
                  local_anchor=bool(args.local_anchor)).to(device)
    if args.ablation == "no_spectral_attention": model.attention.spectral_weight=0.0
    if args.ablation == "no_gp": args.objective="mse"
    if args.ablation == "no_spectral_reg": args.lambda_spectral=0.0
    gp = SpectralGP(2 * args.freq_dim + args.hidden_dim,
                    representation_kernel=bool(args.representation_kernel),
                    base_kernel=args.base_kernel).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(gp.parameters()), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_state_path = args.output_dir / f"{args.dataset}_training_state.pt"
    resume_state = None
    resume_mode = "fresh"
    if args.resume and training_state_path.exists():
        resume_state = torch.load(training_state_path, map_location=device, weights_only=False)
        resume_mode = "full_state"
    elif args.resume:
        # Older completed runs contain the selected model/GP weights but not
        # optimizer momentum.  They remain useful as a warm start and are not
        # rerun merely to upgrade the checkpoint format.
        legacy_path = args.output_dir / f"{args.dataset}_checkpoint.pt"
        legacy_result = args.output_dir / f"{args.dataset}_run.json"
        if legacy_path.exists() and legacy_result.exists():
            legacy = torch.load(legacy_path, map_location=device, weights_only=False)
            legacy_args = legacy.get("args", {})
            completed = int(legacy_args.get("epochs", 0))
            if completed < args.epochs or args.evaluate_only:
                legacy_summary = json.loads(legacy_result.read_text(encoding="utf-8"))
                history_path = args.output_dir / "training_history.json"
                old_history = (json.loads(history_path.read_text(encoding="utf-8"))
                               if history_path.exists() else {})
                resume_state = {
                    "completed_epochs": args.epochs if args.evaluate_only else completed,
                    "model": legacy["model"],
                    "gp": legacy["gp"],
                    "optimizer": optimizer.state_dict(),
                    "training_history": old_history,
                    "best_val_mse": legacy_summary.get("best_val_mse", float("inf")),
                    "best_epoch": legacy_summary.get("best_epoch", completed),
                    "best_model": legacy["model"],
                    "best_gp": legacy["gp"],
                }
                resume_mode = "legacy_weight_warm_start"
    if args.evaluate_only and resume_state is None:
        raise FileNotFoundError(
            f"--evaluate-only requires {training_state_path.name} or a legacy selected checkpoint")
    history = train_sfnp_gp(model, gp, loaders["train"], optimizer, args.epochs, device,
                            args.lambda_gp, args.lambda_mse, args.lambda_spectral, args.objective,
                            val_loader=loaders["val"], restore_best=bool(args.restore_best),
                            checkpoint_path=training_state_path,
                            checkpoint_every=args.checkpoint_every,
                            resume_state=resume_state)
    @torch.no_grad()
    def encode(loader):
        model.eval(); mus=[]; gammas=[]; ys=[]
        for b in loader:
            out=model(b["coord"].to(device), b["aux"].to(device), b["neigh_coord"].to(device),
                      b["neigh_aux"].to(device), b["neigh_target"].to(device))
            mus.append(out["mu"].squeeze(-1)); gammas.append(out["kernel_feature"]); ys.append(b["target"].to(device))
        return torch.cat(mus), torch.cat(gammas), torch.cat(ys)
    train_eval = DataLoader(train, args.batch_size, shuffle=False)
    mu_tr, ga_tr, y_tr = encode(train_eval); mu_va, ga_va, y_va = encode(loaders["val"]); mu_te, ga_te, y_te = encode(loaders["test"])
    base_tr = torch.as_tensor(np.c_[S[itr], X[itr]], dtype=torch.float32, device=device)
    base_va = torch.as_tensor(np.c_[S[iva], X[iva]], dtype=torch.float32, device=device)
    base_te = torch.as_tensor(np.c_[S[ite], X[ite]], dtype=torch.float32, device=device)
    # Mini-batch GP likelihoods are scalable but can bias global length-scales.
    # For moderate samples, refine only covariance hyperparameters against the
    # full frozen training residual vector before evaluating the posterior.
    if args.ablation != "no_gp" and len(ga_tr) <= 2000 and args.gp_refine_steps > 0:
        for parameter in gp.kernel_net.parameters():
            parameter.requires_grad_(False)
        scalar_parameters = [gp.log_sigma_f, gp.log_sigma_coord,
                             gp.log_lengthscale, gp.log_coord_lengthscale,
                             gp.log_noise]
        gp_optimizer = torch.optim.Adam(scalar_parameters, lr=0.03)
        gp.train()
        frozen_residual = (y_tr - mu_tr).detach()
        for _ in range(args.gp_refine_steps):
            gp_optimizer.zero_grad()
            refine_loss = gp.nll(frozen_residual, ga_tr.detach(), base_tr)
            refine_loss.backward()
            gp_optimizer.step()
        for parameter in gp.kernel_net.parameters():
            parameter.requires_grad_(True)
    spectral_weight = torch.tensor(1.0, device=device)
    if args.ablation == "no_gp":
        gp_mean_va=torch.zeros_like(y_va); gp_mean=torch.zeros_like(y_te)
        base_var=torch.var(y_va-mu_va).clamp_min(1e-6); gp_var_va=torch.full_like(y_va,base_var); gp_var=torch.full_like(y_te,base_var)
    else:
        use_nngp = args.gp_inference == "nngp" or (args.gp_inference == "auto" and len(ga_tr) > 2000)
        predictor = gp.predict_nearest_neighbor if use_nngp else gp.predict
        kwargs = {"neighbors": args.gp_neighbors} if use_nngp else {}
        gp_mean_va, gp_var_va = predictor(ga_tr, ga_va, y_tr-mu_tr,
                                          coords_train=base_tr, coords_test=base_va, **kwargs)
        gp_mean, gp_var = predictor(ga_tr, ga_te, y_tr-mu_tr,
                                    coords_train=base_tr, coords_test=base_te, **kwargs)
        if args.residual_ensemble and len(ga_tr) <= 2000:
            # A ridge-trend regression-kriging expert with a geographic
            # Matérn residual protects the model when a low-dimensional
            # stationary covariance is already well matched. Its convex
            # weight is selected on validation responses; test responses
            # remain untouched.
            def matern32(left, right, lengthscale):
                scaled_distance = (3.0 ** 0.5) * torch.cdist(left, right) / lengthscale
                return (1.0 + scaled_distance) * torch.exp(-scaled_distance)

            def stable_cholesky(signal, diagonal_noise, identity):
                """Cholesky with deterministic jitter escalation for duplicates."""
                for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                    chol, info = torch.linalg.cholesky_ex(
                        signal + (diagonal_noise + jitter) * identity)
                    if int(info.max()) == 0:
                        return chol
                raise RuntimeError("Residual Matérn covariance remained non-positive-definite after jitter escalation")

            # The safeguard is a geographic residual GP.  Its Matérn distance
            # must therefore be computed in standardized coordinate space,
            # rather than in a concatenated coordinate--covariate space whose
            # attribute dimensions can distort geographic proximity.
            rk_tr = torch.as_tensor(S[itr], dtype=torch.float32, device=device)
            rk_va = torch.as_tensor(S[iva], dtype=torch.float32, device=device)
            rk_te = torch.as_tensor(S[ite], dtype=torch.float32, device=device)
            # Fit the regression part explicitly; the previous implementation
            # used only a scaled neural mean and therefore was not actually a
            # regression-kriging safeguard.  Ridge strength is chosen on the
            # validation split, using training observations only for fitting.
            design_tr = torch.cat([torch.ones((len(base_tr), 1), device=device), base_tr], dim=1)
            design_va = torch.cat([torch.ones((len(base_va), 1), device=device), base_va], dim=1)
            design_te = torch.cat([torch.ones((len(base_te), 1), device=device), base_te], dim=1)
            ridge_choice = None
            for ridge_alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
                penalty = torch.eye(design_tr.shape[1], device=device)
                penalty[0, 0] = 0.0
                coefficient = torch.linalg.solve(
                    design_tr.T @ design_tr + ridge_alpha * penalty,
                    design_tr.T @ y_tr)
                candidate_trend_va = design_va @ coefficient
                candidate_error = torch.mean((y_va - candidate_trend_va).square())
                if ridge_choice is None or candidate_error < ridge_choice[0]:
                    ridge_choice = (candidate_error, ridge_alpha, coefficient)
            _, rk_ridge_alpha, rk_coefficient = ridge_choice
            rk_trend_tr = design_tr @ rk_coefficient
            rk_trend_va = design_va @ rk_coefficient
            rk_trend_te = design_te @ rk_coefficient

            best = None
            eye = torch.eye(len(rk_tr), device=device, dtype=rk_tr.dtype)
            # Small validation-only grid avoids unstable native optimizers and
            # makes the regression-kriging tuning budget explicit.
            for lengthscale in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5,
                                2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
                k_train_signal = matern32(rk_tr, rk_tr, lengthscale)
                k_val = matern32(rk_va, rk_tr, lengthscale)
                for noise in (0.0001, 0.0005, 0.001, 0.002, 0.005,
                              0.01, 0.02, 0.05, 0.1, 0.2):
                    chol = stable_cholesky(k_train_signal, noise, eye)
                    residual_train = (y_tr - rk_trend_tr).detach()
                    alpha = torch.cholesky_solve(residual_train[:, None], chol).squeeze(-1)
                    candidate_residual_va = k_val @ alpha
                    candidate_error = torch.mean(
                        (y_va - (rk_trend_va + candidate_residual_va)).square())
                    if best is None or candidate_error < best[0]:
                        best = (candidate_error, lengthscale, noise,
                                chol, alpha, candidate_residual_va)
            _, rk_lengthscale, rk_noise, rk_chol, rk_alpha, rk_residual_va = best
            k_test = matern32(rk_te, rk_tr, rk_lengthscale)
            rk_residual_te = k_test @ rk_alpha
            # Corrections are expressed relative to the neural mean used by
            # the common prediction path below.
            rk_mean_va = rk_trend_va + rk_residual_va - mu_va
            rk_mean_te = rk_trend_te + rk_residual_te - mu_te
            rk_solved_va = torch.cholesky_solve(matern32(rk_tr, rk_va, rk_lengthscale), rk_chol)
            rk_solved_te = torch.cholesky_solve(k_test.T, rk_chol)
            rk_var_va = (1.0 + rk_noise - (matern32(rk_va, rk_tr, rk_lengthscale) * rk_solved_va.T).sum(-1)).clamp_min(1e-8)
            rk_var_te = (1.0 + rk_noise - (k_test * rk_solved_te.T).sum(-1)).clamp_min(1e-8)
            weights = torch.linspace(0.0, 1.0, 21, device=device)
            validation_errors = torch.stack([
                torch.mean((y_va - (mu_va + w * gp_mean_va + (1.0 - w) * rk_mean_va)).square())
                for w in weights
            ])
            spectral_weight = weights[torch.argmin(validation_errors)]
            gp_mean_va = spectral_weight * gp_mean_va + (1.0 - spectral_weight) * rk_mean_va
            gp_var_va = spectral_weight.square() * gp_var_va + (1.0 - spectral_weight).square() * rk_var_va
            gp_mean = spectral_weight * gp_mean + (1.0 - spectral_weight) * rk_mean_te
            gp_var = spectral_weight.square() * gp_var + (1.0 - spectral_weight).square() * rk_var_te

    local_expert_weight = torch.tensor(1.0, device=device)
    gp_correction_weight = torch.tensor(1.0, device=device)
    local_k = None
    local_power = None
    local_beta = None
    # Moderate data use exact all-pairs interpolation. Full-data inference uses
    # a KDTree candidate search followed by validation-selected spatial--
    # covariate re-ranking, so no dense query-by-training matrix is formed.
    if args.local_expert and args.ablation != "no_gp" and len(ga_tr) <= 2000:
        coord_tr = torch.as_tensor(S[itr], dtype=torch.float32, device=device)
        coord_va = torch.as_tensor(S[iva], dtype=torch.float32, device=device)
        coord_te = torch.as_tensor(S[ite], dtype=torch.float32, device=device)

        def local_predict(query, k, power):
            distance = torch.cdist(query, coord_tr)
            selected_distance, selected_index = torch.topk(
                distance, k=min(k, len(coord_tr)), largest=False)
            selected_y = y_tr[selected_index]
            weight = (selected_distance + 1e-4).pow(-power)
            weight = weight / weight.sum(dim=1, keepdim=True)
            mean = (weight * selected_y).sum(dim=1)
            variance = (weight * (selected_y - mean[:, None]).square()).sum(dim=1).clamp_min(1e-4)
            return mean, variance

        best_local = None
        for candidate_k in (8, 16, 32, 64):
            for candidate_power in (0.5, 1.0, 2.0, 3.0):
                local_mean_va, local_var_va = local_predict(
                    coord_va, candidate_k, candidate_power)
                for expert_weight in torch.linspace(0.0, 1.0, 21, device=device):
                    candidate_prediction = (
                        expert_weight * (mu_va + gp_mean_va)
                        + (1.0 - expert_weight) * local_mean_va)
                    candidate_error = torch.mean((y_va - candidate_prediction).square())
                    if best_local is None or candidate_error < best_local[0]:
                        best_local = (candidate_error, candidate_k, candidate_power,
                                      expert_weight, local_mean_va, local_var_va)
        _, local_k, local_power, local_expert_weight, local_mean_va, local_var_va = best_local
        local_mean_te, local_var_te = local_predict(coord_te, local_k, local_power)
        base_prediction_va = mu_va + gp_mean_va
        base_prediction_te = mu_te + gp_mean
        combined_prediction_va = (local_expert_weight * base_prediction_va
                                  + (1.0 - local_expert_weight) * local_mean_va)
        combined_prediction_te = (local_expert_weight * base_prediction_te
                                  + (1.0 - local_expert_weight) * local_mean_te)
        gp_mean_va = combined_prediction_va - mu_va
        gp_mean = combined_prediction_te - mu_te
    covariate_expert_weight = torch.tensor(1.0, device=device)
    covariate_tree_leaf = None
    covariate_tree_features = None
    covariate_residual_weight = None
    covariate_lengthscale = None
    covariate_noise = None
    if args.covariate_expert and len(ga_tr) <= 2000:
        # Some soil responses are primarily controlled by measured site
        # attributes.  A training-only tree trend is therefore offered as a
        # transparent safeguard, while a geographic Matérn correction retains
        # residual spatial structure.  Every discrete choice and the final
        # convex mixing weight are selected on validation responses only.
        covariate_tr = np.c_[S[itr], X[itr]]
        covariate_va = np.c_[S[iva], X[iva]]
        covariate_te = np.c_[S[ite], X[ite]]
        tree_choice = None
        for candidate_leaf in (1, 2, 4, 8):
            for candidate_features in (0.7, 1.0):
                tree_model = ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=candidate_leaf,
                    max_features=candidate_features,
                    n_jobs=-1,
                    random_state=args.seed,
                ).fit(covariate_tr, y[itr])
                tree_va = tree_model.predict(covariate_va)
                tree_error = np.mean((y[iva] - tree_va) ** 2)
                if tree_choice is None or tree_error < tree_choice[0]:
                    tree_choice = (tree_error, candidate_leaf,
                                   candidate_features, tree_model, tree_va)
        (_, covariate_tree_leaf, covariate_tree_features,
         tree_model, tree_va_np) = tree_choice
        tree_tr = torch.as_tensor(
            tree_model.predict(covariate_tr), dtype=y_tr.dtype, device=device)
        tree_va = torch.as_tensor(tree_va_np, dtype=y_va.dtype, device=device)
        tree_te = torch.as_tensor(
            tree_model.predict(covariate_te), dtype=y_te.dtype, device=device)

        coord_tr = torch.as_tensor(S[itr], dtype=torch.float32, device=device)
        coord_va = torch.as_tensor(S[iva], dtype=torch.float32, device=device)
        coord_te = torch.as_tensor(S[ite], dtype=torch.float32, device=device)

        def covariate_matern32(left, right, lengthscale):
            scaled = np.sqrt(3.0) * torch.cdist(left, right) / lengthscale
            return (1.0 + scaled) * torch.exp(-scaled)

        identity = torch.eye(len(coord_tr), dtype=coord_tr.dtype, device=device)
        tree_residual = (y_tr - tree_tr).detach()
        covariate_gp_choice = None
        for candidate_lengthscale in (0.1, 0.2, 0.35, 0.5, 0.75,
                                      1.0, 1.5, 2.0, 3.0, 4.0):
            signal = covariate_matern32(
                coord_tr, coord_tr, candidate_lengthscale)
            cross_va = covariate_matern32(
                coord_va, coord_tr, candidate_lengthscale)
            for candidate_noise in (0.001, 0.005, 0.01, 0.02,
                                    0.05, 0.1, 0.2):
                chol = None
                for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                    candidate_chol, info = torch.linalg.cholesky_ex(
                        signal + (candidate_noise + jitter) * identity)
                    if int(info.max()) == 0:
                        chol = candidate_chol
                        break
                if chol is None:
                    continue
                alpha = torch.cholesky_solve(
                    tree_residual[:, None], chol).squeeze(-1)
                residual_va = cross_va @ alpha
                for residual_weight in torch.linspace(
                        0.0, 1.0, 5, device=device):
                    candidate_prediction = tree_va + residual_weight * residual_va
                    candidate_error = torch.mean(
                        (y_va - candidate_prediction).square())
                    if (covariate_gp_choice is None
                            or candidate_error < covariate_gp_choice[0]):
                        covariate_gp_choice = (
                            candidate_error, candidate_lengthscale,
                            candidate_noise, residual_weight, alpha,
                            candidate_prediction)
        (_, covariate_lengthscale, covariate_noise,
         covariate_residual_weight, covariate_alpha,
         covariate_prediction_va) = covariate_gp_choice
        covariate_residual_te = covariate_matern32(
            coord_te, coord_tr, covariate_lengthscale) @ covariate_alpha
        covariate_prediction_te = (
            tree_te + covariate_residual_weight * covariate_residual_te)

        base_prediction_va = mu_va + gp_mean_va
        base_prediction_te = mu_te + gp_mean
        route_weights = torch.linspace(0.0, 1.0, 21, device=device)
        route_errors = torch.stack([
            torch.mean((y_va - (weight * base_prediction_va
                                + (1.0 - weight)
                                * covariate_prediction_va)).square())
            for weight in route_weights
        ])
        covariate_expert_weight = route_weights[torch.argmin(route_errors)]
        combined_prediction_va = (
            covariate_expert_weight * base_prediction_va
            + (1.0 - covariate_expert_weight) * covariate_prediction_va)
        combined_prediction_te = (
            covariate_expert_weight * base_prediction_te
            + (1.0 - covariate_expert_weight) * covariate_prediction_te)
        gp_mean_va = combined_prediction_va - mu_va
        gp_mean = combined_prediction_te - mu_te
        # The local expert adjusts the conditional mean only.  Uncertainty is
        # still propagated by the residual GP and then globally calibrated on
        # validation residuals; neighbour-response dispersion is not treated
        # as an independent Gaussian variance component.
    if (args.local_expert and args.ablation != "no_gp"
            and len(ga_tr) > 2000):
        candidate_count = min(256, len(itr))
        tree = KDTree(S[itr])

        def scalable_local_candidates(query_indices):
            spatial_distance, candidate_indices = tree.query(
                S[query_indices], k=candidate_count)
            neighbour_x = X[itr][candidate_indices]
            covariate_distance = np.linalg.norm(
                neighbour_x - X[query_indices, None, :], axis=-1
            ) / np.sqrt(X.shape[1])
            neighbour_y = y[itr][candidate_indices]
            for beta in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0):
                hybrid_distance = spatial_distance + beta * covariate_distance
                for candidate_k in (8, 16, 32, 64, 128):
                    selected = np.argpartition(
                        hybrid_distance, kth=candidate_k - 1, axis=1)[:, :candidate_k]
                    selected_distance = np.take_along_axis(
                        hybrid_distance, selected, axis=1)
                    selected_y = np.take_along_axis(neighbour_y, selected, axis=1)
                    for candidate_power in (0.5, 1.0, 2.0, 3.0):
                        weight = np.power(selected_distance + 1e-4, -candidate_power)
                        prediction = np.sum(weight * selected_y, axis=1) / np.sum(weight, axis=1)
                        yield beta, candidate_k, candidate_power, prediction

        def scalable_local_predict(query_indices, beta, k, power):
            spatial_distance, candidate_indices = tree.query(
                S[query_indices], k=candidate_count)
            neighbour_x = X[itr][candidate_indices]
            covariate_distance = np.linalg.norm(
                neighbour_x - X[query_indices, None, :], axis=-1
            ) / np.sqrt(X.shape[1])
            hybrid_distance = spatial_distance + beta * covariate_distance
            selected = np.argpartition(hybrid_distance, kth=k - 1, axis=1)[:, :k]
            selected_distance = np.take_along_axis(hybrid_distance, selected, axis=1)
            selected_y = np.take_along_axis(y[itr][candidate_indices], selected, axis=1)
            weight = np.power(selected_distance + 1e-4, -power)
            return np.sum(weight * selected_y, axis=1) / np.sum(weight, axis=1)

        # Both weights and all local-expert parameters are selected only on the
        # validation partition. The spectral-route lower bound prevents the
        # fitted ASDKL representation from being replaced by the safeguard.
        best_large_local = None
        for beta, candidate_k, candidate_power, local_prediction in scalable_local_candidates(iva):
            local_prediction = torch.as_tensor(
                local_prediction, dtype=y_va.dtype, device=device)
            for correction_weight in torch.linspace(0.0, 1.5, 13, device=device):
                spectral_prediction = mu_va + correction_weight * gp_mean_va
                for expert_weight in torch.linspace(0.5, 1.0, 11, device=device):
                    candidate_prediction = (
                        expert_weight * spectral_prediction
                        + (1.0 - expert_weight) * local_prediction)
                    candidate_error = torch.mean((y_va - candidate_prediction).square())
                    if best_large_local is None or candidate_error < best_large_local[0]:
                        best_large_local = (
                            candidate_error, beta, candidate_k, candidate_power,
                            correction_weight, expert_weight)
        (_, local_beta, local_k, local_power,
         gp_correction_weight, local_expert_weight) = best_large_local
        local_mean_va = torch.as_tensor(
            scalable_local_predict(iva, local_beta, local_k, local_power),
            dtype=y_va.dtype, device=device)
        local_mean_te = torch.as_tensor(
            scalable_local_predict(ite, local_beta, local_k, local_power),
            dtype=y_te.dtype, device=device)
        combined_prediction_va = (
            local_expert_weight * (mu_va + gp_correction_weight * gp_mean_va)
            + (1.0 - local_expert_weight) * local_mean_va)
        combined_prediction_te = (
            local_expert_weight * (mu_te + gp_correction_weight * gp_mean)
            + (1.0 - local_expert_weight) * local_mean_te)
        gp_mean_va = combined_prediction_va - mu_va
        gp_mean = combined_prediction_te - mu_te
    # A single validation-only scale calibrates Gaussian dispersion without
    # touching test responses.  For a global multiplicative scale, the value
    # below is the closed-form minimizer of validation Gaussian NLL.
    gaussian_z = 1.6448536269514722
    calibration_ratios = ((y_va - (mu_va + gp_mean_va)).square()
                          / gp_var_va.clamp_min(1e-8))
    calibration = torch.sqrt(calibration_ratios.mean()).item()
    if len(y_va) < 50:
        # With only 20 validation observations, use the finite-sample
        # split-conformal order statistic for interval coverage.  Gaussian
        # NLL and CRPS retain the validation-NLL variance scale above.
        standardized_scores = (torch.abs(y_va - (mu_va + gp_mean_va))
                               / (torch.sqrt(gp_var_va.clamp_min(1e-8))
                                  * max(calibration, 1e-3)))
        conformal_rank = min(len(standardized_scores),
                             int(np.ceil((len(standardized_scores) + 1) * 0.90)))
        interval_z = float(torch.kthvalue(standardized_scores, conformal_rank).values)
    else:
        interval_z = gaussian_z
    gp_var = gp_var * max(calibration, 1e-3) ** 2
    pred = mu_te + gp_mean
    target_scale = float(scalers[2].scale_[0]); target_mean = float(scalers[2].mean_[0])
    pred_np = (pred.cpu().numpy()*target_scale+target_mean); y_np=(y_te.cpu().numpy()*target_scale+target_mean)
    var_np = gp_var.cpu().numpy()*target_scale**2 + 1e-8
    predictive_std_np = np.sqrt(var_np)
    standardized_error = (y_np - pred_np) / predictive_std_np
    gaussian_crps = predictive_std_np * (
        standardized_error * (2 * ndtr(standardized_error) - 1)
        + 2 * np.exp(-0.5 * standardized_error**2) / np.sqrt(2 * np.pi)
        - 1 / np.sqrt(np.pi)
    )
    va_pred_np=((mu_va+gp_mean_va).cpu().numpy()*target_scale+target_mean)
    va_y_np=(y_va.cpu().numpy()*target_scale+target_mean)
    metrics = {"val_rmse": float(mean_squared_error(va_y_np,va_pred_np)**0.5),
               "rmse": float(mean_squared_error(y_np,pred_np)**0.5),
               "mae": float(mean_absolute_error(y_np,pred_np)), "r2": float(r2_score(y_np,pred_np)),
               "nll": float(np.mean(0.5*np.log(2*np.pi*var_np)+(y_np-pred_np)**2/(2*var_np))),
               "crps": float(np.mean(gaussian_crps)),
               "picp90": float(np.mean((y_np>=pred_np-interval_z*np.sqrt(var_np))&(y_np<=pred_np+interval_z*np.sqrt(var_np)))),
               "mpiw90": float(np.mean(2*interval_z*np.sqrt(var_np))),
               "interval_scale": float(calibration), "interval_quantile": float(interval_z),
               "spectral_residual_weight": float(spectral_weight),
               "rk_lengthscale": float(rk_lengthscale) if args.residual_ensemble and args.ablation != "no_gp" and len(ga_tr) <= 2000 else None,
               "rk_noise": float(rk_noise) if args.residual_ensemble and args.ablation != "no_gp" and len(ga_tr) <= 2000 else None,
               "rk_ridge_alpha": float(rk_ridge_alpha) if args.residual_ensemble and args.ablation != "no_gp" and len(ga_tr) <= 2000 else None,
               "gp_correction_weight": float(gp_correction_weight),
               "local_expert_weight": float(local_expert_weight),
               "covariate_expert_weight": float(covariate_expert_weight),
               "covariate_tree_leaf": covariate_tree_leaf,
               "covariate_tree_features": covariate_tree_features,
               "covariate_residual_weight": float(covariate_residual_weight) if covariate_residual_weight is not None else None,
               "covariate_lengthscale": covariate_lengthscale,
               "covariate_noise": covariate_noise,
               "local_beta": local_beta,
               "local_k": local_k, "local_power": local_power}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history["training_history"], indent=2), encoding="utf-8")
    # Source data for prediction, residual and uncertainty figures. Coordinates
    # are restored to their original units and test targets are never used for
    # fitting or interval calibration.
    coord_raw = scalers[0].inverse_transform(S[ite])
    np.savez_compressed(args.output_dir / f"{args.dataset}_predictions.npz",
                        coord=coord_raw, target=y_np, prediction=pred_np,
                        predictive_std=np.sqrt(var_np), test_index=ite)
    torch.save({"model": model.state_dict(), "gp": gp.state_dict(), "args": vars(args)}, args.output_dir / f"{args.dataset}_checkpoint.pt")
    estimator_name = ("ASDKL-Covariate"
                      if args.covariate_expert else "ASDKL")
    summary = {"dataset": args.dataset, "estimator": estimator_name,
               "seed": args.seed, "ablation":args.ablation,
               "objective": args.objective,
               "checkpoint_selection": "minimum validation MSE" if args.restore_best else "final epoch",
               "resume_mode": resume_mode,
               "gp_inference": "predictive_local" if (args.gp_inference == "nngp" or (args.gp_inference == "auto" and len(itr) > 2000)) else "exact",
               "n": len(y), "train": len(itr), "validation": len(iva), "test": len(ite),
               "test_fraction": args.test_size, "validation_fraction": args.val_size,
               **history, **metrics}
    (args.output_dir / f"{args.dataset}_run.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    concise_keys = ("dataset", "seed", "ablation", "objective", "val_rmse", "rmse", "mae", "r2", "nll", "crps", "picp90", "mpiw90")
    print(json.dumps({key: summary[key] for key in concise_keys}, default=str))


if __name__ == "__main__": main()



