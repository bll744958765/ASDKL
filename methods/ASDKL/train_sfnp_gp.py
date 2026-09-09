import copy
from pathlib import Path

import torch

try:
    from .loss import SFNPLoss
except ImportError:  # Direct execution from methods/ASDKL.
    from loss import SFNPLoss


def _save_training_state(path, epoch, model, gp_model, optimizer, history,
                         best_val_mse, best_epoch, best_model_state,
                         best_gp_state):
    """Atomically save everything required to continue the next epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "completed_epochs": int(epoch),
        "model": model.state_dict(),
        "gp": gp_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_history": history,
        "best_val_mse": float(best_val_mse),
        "best_epoch": int(best_epoch),
        "best_model": best_model_state,
        "best_gp": best_gp_state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_sfnp_gp(

        model,

        gp_model,

        train_loader,

        optimizer,

        epochs,

        device,

        lambda_gp=1.0,

        lambda_mse=0.0,

        lambda_spectral=0.01,

        objective="joint_nll",
        val_loader=None,
        restore_best=False,
        checkpoint_path=None,
        checkpoint_every=25,
        resume_state=None):

    criterion = SFNPLoss(

        lambda_gp=lambda_gp,

        lambda_mse=lambda_mse,

        lambda_spectral=lambda_spectral,

        objective=objective

    )

    model.train()

    gp_model.train()

    history = {"epoch": [], "total": [], "mse": [], "nll": [],
               "spectral": [], "val_total": [], "val_mse": []}
    best_val_mse = float("inf")
    best_epoch = 0
    best_model_state = None
    best_gp_state = None
    start_epoch = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        gp_model.load_state_dict(resume_state["gp"])
        optimizer.load_state_dict(resume_state["optimizer"])
        history = resume_state.get("training_history", history)
        best_val_mse = float(resume_state.get("best_val_mse", best_val_mse))
        best_epoch = int(resume_state.get("best_epoch", best_epoch))
        best_model_state = resume_state.get("best_model")
        best_gp_state = resume_state.get("best_gp")
        start_epoch = int(resume_state.get("completed_epochs", 0))
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        cuda_state = resume_state.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    if start_epoch > epochs:
        raise ValueError(f"Checkpoint already contains {start_epoch} epochs, exceeding requested {epochs}.")
    for epoch in range(start_epoch, epochs):

        epoch_loss = 0
        epoch_parts = {"pred": 0.0, "gp": 0.0, "spectral": 0.0}

        loop = train_loader

        for batch in loop:

            s = batch["coord"].to(device)

            x = batch["aux"].to(device)

            y = batch["target"].to(device)

            neigh_s = batch["neigh_coord"].to(device)

            neigh_x = batch["neigh_aux"].to(device)

            neigh_y = batch["neigh_target"].to(device)

            optimizer.zero_grad()

            ################################################
            # SFNP
            ################################################

            outputs = model(

                s,

                x,

                neigh_s,

                neigh_x,

                neigh_y

            )

            mu = outputs["mu"].squeeze()

            gamma = outputs["kernel_feature"]

            omega = outputs["omega"]

            ################################################
            # GP
            ################################################

            K, latent_gp = gp_model(gamma, torch.cat([s, x], dim=-1))

            residual = y - mu

            ################################################
            # Loss
            ################################################

            losses = criterion(

                y_true=y,

                y_pred=mu,

                omega=omega,

                coords=s,

                residual=residual,

                K=K

            )

            losses["total"].backward()

            optimizer.step()

            epoch_loss += (

                losses["total"].item()

            )
            for key in epoch_parts:
                epoch_parts[key] += float(losses[key].detach().item())

        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(f"Epoch={epoch+1} Loss={epoch_loss:.4f}")
        batches = max(1, len(train_loader))
        history["epoch"].append(epoch + 1)
        history["total"].append(epoch_loss / batches)
        history["mse"].append(epoch_parts["pred"] / batches)
        history["nll"].append(epoch_parts["gp"] / batches)
        history["spectral"].append(epoch_parts["spectral"] / batches)

        # Validation objective is diagnostic and never back-propagated.  It
        # distinguishes optimization convergence from continued overfitting.
        val_total = float("nan")
        val_mse = float("nan")
        # Sparse-label runs restore the validation-selected checkpoint.  In
        # that regime every epoch is evaluated because 80 fitting examples
        # can overfit between the coarser diagnostics used for dense runs.
        # Test responses are never inspected during this selection.
        validation_stride = 1 if restore_best else max(1, epochs // 30)
        if val_loader is not None and ((epoch + 1) % validation_stride == 0 or epoch == 0 or epoch + 1 == epochs):
            model.eval(); gp_model.eval(); values = []; mse_values = []
            with torch.no_grad():
                for batch in val_loader:
                    s = batch["coord"].to(device); x = batch["aux"].to(device)
                    y = batch["target"].to(device)
                    out = model(s, x, batch["neigh_coord"].to(device),
                                batch["neigh_aux"].to(device),
                                batch["neigh_target"].to(device))
                    mu = out["mu"].squeeze(); K, _ = gp_model(
                        out["kernel_feature"], torch.cat([s, x], dim=-1))
                    parts = criterion(y_true=y, y_pred=mu, omega=out["omega"],
                                      coords=s, residual=y-mu, K=K)
                    values.append(float(parts["total"].item()))
                    mse_values.append(float(parts["pred"].item()))
            val_total = sum(values) / max(1, len(values))
            val_mse = sum(mse_values) / max(1, len(mse_values))
            # Model selection uses only held-out validation responses.  MSE is
            # not added to the training objective; it is used here because it
            # directly selects the mean representation before GP posterior
            # construction and is not confounded by batch covariance size.
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                best_gp_state = copy.deepcopy(gp_model.state_dict())
            model.train(); gp_model.train()
        history["val_total"].append(val_total)
        history["val_mse"].append(val_mse)

        if checkpoint_path is not None and (
                (epoch + 1) % max(1, checkpoint_every) == 0 or epoch + 1 == epochs):
            _save_training_state(
                checkpoint_path, epoch + 1, model, gp_model, optimizer, history,
                best_val_mse, best_epoch, best_model_state, best_gp_state)

    if restore_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        gp_model.load_state_dict(best_gp_state)
    final_loss = history["total"][-1] if history["total"] else float("nan")
    return {"loss": final_loss, "trained_epochs": epochs,
            "best_epoch": best_epoch, "best_val_mse": best_val_mse,
            "training_history": history}
