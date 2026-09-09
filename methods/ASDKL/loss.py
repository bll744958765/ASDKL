import torch
import torch.nn as nn


class SFNPLoss(nn.Module):

    def __init__(

            self,

            lambda_gp=1.0,

            lambda_mse=0.0,

            lambda_spectral=0.01,

            objective="joint_nll"):

        super().__init__()

        self.lambda_gp = lambda_gp

        self.lambda_mse = lambda_mse

        self.lambda_spectral = lambda_spectral

        if objective not in {"joint_nll", "hybrid_mse_nll", "mse"}:
            raise ValueError(f"Unknown objective: {objective}")
        self.objective = objective

    ####################################################
    # Prediction Loss
    ####################################################

    def prediction_loss(

            self,

            y_true,

            y_pred):

        return torch.mean(

            (y_true - y_pred) ** 2

        )

    ####################################################
    # GP Marginal Likelihood
    ####################################################

    def gp_loss(

            self,

            residual,

            K):

        N = residual.shape[0]

        jitter = 1e-6

        K = K + jitter * torch.eye(

            N,

            device=K.device

        )

        L = torch.linalg.cholesky(K)

        alpha = torch.cholesky_solve(

            residual.unsqueeze(-1),

            L

        )

        data_fit = (

            0.5
            *
            residual.unsqueeze(0)
            @
            alpha

        ).squeeze()

        logdet = (

            torch.log(

                torch.diag(L)

            ).sum()
            * 2
        )

        complexity = 0.5 * logdet

        constant = (

            0.5
            *
            N
            *
            torch.log(

                torch.tensor(
                    2.0 * torch.pi,
                    device=K.device
                )
            )
        )

        return (

            data_fit
            +
            complexity
            +
            constant

        ) / N

    ####################################################
    # Spectral Smoothness
    ####################################################

    def spectral_loss(

            self,

            omega,

            coords,

            sigma=0.1):

        """
        omega:
        [B,M,2]

        coords:
        [B,2]
        """

        B = coords.shape[0]

        dist = torch.cdist(

            coords,

            coords

        )

        W = torch.exp(

            -dist ** 2
            /
            (
                2
                *
                sigma ** 2
            )
        )

        # Self-pairs have zero frequency distance and previously dominated the
        # batch mean when few off-diagonal locations were within the spatial
        # bandwidth.  Remove them and normalize by effective neighbour mass.
        W.fill_diagonal_(0.0)

        omega_flat = omega.reshape(

            B,

            -1

        )

        freq_dist = torch.cdist(

            omega_flat,

            omega_flat

        )

        loss = (W * freq_dist ** 2).sum() / W.sum().clamp_min(1e-8)

        return loss

    ####################################################
    # Total
    ####################################################

    def forward(

            self,

            y_true,

            y_pred,

            omega,

            coords,

            residual,

            K):

        L_pred = self.prediction_loss(

            y_true,

            y_pred

        )

        L_gp = self.gp_loss(

            residual,

            K

        )

        L_spec = self.spectral_loss(

            omega,

            coords

        )

        # The default objective has two trainable parts: the Gaussian NLL of
        # y ~ N(mu_theta, K_psi) and spatial smoothness of adaptive frequencies.
        # Its Mahalanobis data-fit term is already a covariance-weighted squared
        # error (and becomes MSE up to scale when K = sigma^2 I), so an extra
        # pointwise MSE is disabled by default.  It remains available only as a
        # declared sensitivity option; the raw MSE is always logged diagnostically.
        if self.objective == "joint_nll":
            total = L_gp + self.lambda_mse * L_pred + self.lambda_spectral * L_spec
        elif self.objective == "hybrid_mse_nll":
            total = L_pred + self.lambda_gp * L_gp + self.lambda_spectral * L_spec
        else:
            total = L_pred + self.lambda_spectral * L_spec

        return {

            "total": total,

            "pred": L_pred,

            "gp": L_gp,

            "spectral": L_spec

        }
