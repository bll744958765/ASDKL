import torch
import torch.nn as nn


class SpectralRBFKernel(nn.Module):

    def __init__(
            self,
            lengthscale=1.0,
            variance=1.0):

        super().__init__()

        self.log_lengthscale = nn.Parameter(
            torch.tensor(lengthscale).log()
        )

        self.log_variance = nn.Parameter(
            torch.tensor(variance).log()
        )

    @property
    def lengthscale(self):

        return torch.exp(
            self.log_lengthscale
        )

    @property
    def variance(self):

        return torch.exp(
            self.log_variance
        )

    def forward(
            self,
            gamma1,
            gamma2):

        diff = (
                gamma1[:, None, :]
                -
                gamma2[None, :, :]
        )

        dist2 = (
            diff ** 2
        ).sum(
            dim=-1
        )

        K = (
                self.variance
                *
                torch.exp(
                    -dist2
                    /
                    (
                            2
                            *
                            self.lengthscale ** 2
                    )
                )
        )

        return K
    

###  Spectral GP  r∼GP(0,Kγ​)

class ExactSpectralGP(nn.Module):
    
    def __init__(
            self,
            noise=1e-2):

        super().__init__()

        self.kernel = (
            SpectralRBFKernel()
        )

        self.log_noise = nn.Parameter(
            torch.tensor(noise).log()
        )

    @property
    def noise(self):

        return torch.exp(
            self.log_noise
        )

    def covariance(
            self,
            gamma):

        K = self.kernel(
            gamma,
            gamma
        )

        K = (
                K
                +
                self.noise
                *
                torch.eye(
                    len(gamma),
                    device=gamma.device
                )
        )

        return K

    def nll(
            self,
            residual,
            gamma):

        K = self.covariance(
            gamma
        )

        L = torch.linalg.cholesky(
            K
        )

        alpha = torch.cholesky_solve(
            residual.unsqueeze(-1),
            L
        )

        term1 = (
            residual.unsqueeze(0)
            @
            alpha
        ).squeeze()

        term2 = (
            2
            *
            torch.log(
                torch.diagonal(L)
            )
        ).sum()

        term3 = (
                len(residual)
                *
                torch.log(
                    torch.tensor(
                        2.0 * torch.pi,
                        device=gamma.device
                    )
                )
        )

        return 0.5 * (
                term1
                +
                term2
                +
                term3
        )

###Prediction y^​∗=μ∗	​+K∗NK−1rVar(y∗)=K∗∗−K∗NK−1KN∗

    @torch.no_grad()
    def predict(

            self,

            gamma_train,

            gamma_test,

            residual_train):

        K = self.covariance(
            gamma_train
        )

        Kstar = self.kernel(
            gamma_test,
            gamma_train
        )

        Kss = self.kernel(
            gamma_test,
            gamma_test
        )

        Kinv = torch.inverse(K)

        mean = (
            Kstar
            @
            Kinv
            @
            residual_train
        )

        cov = (
                Kss
                -
                Kstar
                @
                Kinv
                @
                Kstar.T
        )

        var = torch.diag(
            cov
        )

        return mean, var
    
# Learned representation kernel used by the residual GP.



class KernelNetwork(nn.Module):

    def __init__(
            self,
            gamma_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                gamma_dim,
                128
            ),

            nn.SiLU(),

            nn.Linear(
                128,
                64
            )
        )

    def forward(
            self,
            gamma):

        return self.net(
            gamma
        )


class SpectralGP(nn.Module):

    def __init__(
            self,
            gamma_dim,
            representation_kernel=True,
            base_kernel="rbf"):

        super().__init__()

        if base_kernel not in {"matern32", "rbf", "none"}:
            raise ValueError(f"Unsupported base kernel: {base_kernel}")
        self.representation_kernel = bool(representation_kernel)
        self.base_kernel = base_kernel

        self.kernel_net = KernelNetwork(
            gamma_dim
        )

        self.log_sigma_f = nn.Parameter(
            torch.tensor(0.0)
        )

        self.log_sigma_coord = nn.Parameter(torch.tensor(-0.5))

        self.log_lengthscale = nn.Parameter(
            torch.tensor(0.0)
        )

        self.log_coord_lengthscale = nn.Parameter(torch.tensor(0.0))

        self.log_noise = nn.Parameter(
            torch.tensor(-2.0)
        )

    def kernel(
            self,
            u1,
            u2=None):

        if u2 is None:
            u2 = u1

        sigma_f = torch.exp(
            self.log_sigma_f
        )

        l = torch.exp(
            self.log_lengthscale
        )

        dist = torch.cdist(u1, u2)

        K = (
            sigma_f ** 2
        ) * torch.exp(

            -dist ** 2
            /
            (
                2
                * l ** 2
            )
        )

        return K

    def cross_covariance(self, gamma1, gamma2):
        return self.kernel(self.kernel_net(gamma1), self.kernel_net(gamma2))

    def coordinate_kernel(self, coords1, coords2):
        """Selected covariance on standardized coordinate--covariate vectors."""
        sigma = torch.exp(self.log_sigma_coord)
        lengthscale = torch.exp(self.log_coord_lengthscale).clamp_min(1e-4)
        distance = torch.cdist(coords1, coords2)
        if self.base_kernel == "none":
            return torch.zeros_like(distance)
        if self.base_kernel == "rbf":
            return sigma.square() * torch.exp(-distance.square() / (2.0 * lengthscale.square()))
        scaled = (3.0 ** 0.5) * distance / lengthscale
        return sigma.square() * (1.0 + scaled) * torch.exp(-scaled)

    def combined_cross_covariance(self, gamma1, gamma2, coords1=None, coords2=None):
        if self.representation_kernel:
            covariance = self.cross_covariance(gamma1, gamma2)
        else:
            covariance = gamma1.new_zeros((gamma1.shape[0], gamma2.shape[0]))
        if self.base_kernel != "none" and coords1 is not None and coords2 is not None:
            covariance = covariance + self.coordinate_kernel(coords1, coords2)
        return covariance

    def forward(
            self,
            gamma,
            coords=None):

        """
        gamma:
        [N,G]
        """

        u = self.kernel_net(
            gamma
        )

        if self.representation_kernel:
            K = self.kernel(u)
        else:
            K = gamma.new_zeros((gamma.shape[0], gamma.shape[0]))

        if self.base_kernel != "none" and coords is not None:
            K = K + self.coordinate_kernel(coords, coords)

        noise = torch.exp(
            self.log_noise
        )

        K = K + noise * torch.eye(
            K.shape[0],
            device=K.device
        )

        return K, u

    def nll(self, residual, gamma, coords=None):
        K, _ = self(gamma, coords)
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(len(gamma), device=gamma.device))
        alpha = torch.cholesky_solve(residual[:, None], L)
        return (0.5 * residual[None, :] @ alpha).squeeze() + torch.log(torch.diagonal(L)).sum() + 0.5 * len(gamma) * residual.new_tensor(2.0 * torch.pi).log()

    @torch.no_grad()
    def predict(self, gamma_train, gamma_test, residual_train,
                coords_train=None, coords_test=None):
        K, _ = self(gamma_train, coords_train)
        K = K + 1e-6 * torch.eye(len(gamma_train), device=gamma_train.device)
        Ks = self.combined_cross_covariance(gamma_test, gamma_train,
                                             coords_test, coords_train)
        Kss = self.combined_cross_covariance(gamma_test, gamma_test,
                                              coords_test, coords_test)
        L = torch.linalg.cholesky(K)
        mean = Ks @ torch.cholesky_solve(residual_train[:, None], L)
        solved = torch.cholesky_solve(Ks.T, L)
        # Predictive variance includes observation noise because manuscript
        # intervals target observed responses rather than the latent function.
        var = (torch.diagonal(Kss - Ks @ solved) + torch.exp(self.log_noise)).clamp_min(1e-8)
        return mean.squeeze(-1), var

    @torch.no_grad()
    def predict_nearest_neighbor(self, gamma_train, gamma_test, residual_train,
                                 neighbors=32, chunk_size=256,
                                 coords_train=None, coords_test=None):
        """Local GP posterior using nearest neighbours in learned kernel space.

        This avoids constructing the full training covariance and makes held-out
        inference linear in the number of queries for fixed neighbour count.
        """
        u_train = self.kernel_net(gamma_train)
        u_test = self.kernel_net(gamma_test)
        m = min(int(neighbors), len(u_train))
        sigma2 = torch.exp(self.log_sigma_f).square()
        coord_sigma2 = torch.exp(self.log_sigma_coord).square()
        lengthscale = torch.exp(self.log_lengthscale)
        coord_lengthscale = torch.exp(self.log_coord_lengthscale).clamp_min(1e-4)
        noise = torch.exp(self.log_noise)
        means, variances = [], []
        eye = torch.eye(m, device=u_train.device)
        for start in range(0, len(u_test), chunk_size):
            uq = u_test[start:start + chunk_size]
            distances = torch.zeros(
                (len(uq), len(u_train)), device=u_train.device,
                dtype=u_train.dtype)
            if self.representation_kernel:
                distances = distances + torch.cdist(
                    uq / lengthscale, u_train / lengthscale)
            if (self.base_kernel != "none" and coords_train is not None
                    and coords_test is not None):
                cq = coords_test[start:start + chunk_size]
                distances = distances + torch.cdist(cq / coord_lengthscale,
                                                     coords_train / coord_lengthscale)
            indices = torch.topk(distances, k=m, largest=False).indices
            un = u_train[indices]
            rn = residual_train[indices]
            pair_dist2 = (un[:, :, None, :] - un[:, None, :, :]).square().sum(-1)
            if self.representation_kernel:
                knn = sigma2 * torch.exp(
                    -pair_dist2 / (2 * lengthscale.square()))
            else:
                knn = pair_dist2.new_zeros(pair_dist2.shape)
            if (self.base_kernel != "none" and coords_train is not None
                    and coords_test is not None):
                cn = coords_train[indices]
                coord_pair = torch.linalg.vector_norm(
                    cn[:, :, None, :] - cn[:, None, :, :], dim=-1)
                if self.base_kernel == "rbf":
                    knn = knn + coord_sigma2 * torch.exp(
                        -coord_pair.square() / (2.0 * coord_lengthscale.square()))
                else:
                    scaled_pair = (3.0 ** 0.5) * coord_pair / coord_lengthscale
                    knn = knn + coord_sigma2 * (1.0 + scaled_pair) * torch.exp(-scaled_pair)
            knn = knn + (noise + 1e-6) * eye.unsqueeze(0)
            cross_dist2 = (uq[:, None, :] - un).square().sum(-1)
            if self.representation_kernel:
                kstar = sigma2 * torch.exp(
                    -cross_dist2 / (2 * lengthscale.square()))
            else:
                kstar = cross_dist2.new_zeros(cross_dist2.shape)
            if (self.base_kernel != "none" and coords_train is not None
                    and coords_test is not None):
                coord_cross = torch.linalg.vector_norm(cq[:, None, :] - cn, dim=-1)
                if self.base_kernel == "rbf":
                    kstar = kstar + coord_sigma2 * torch.exp(
                        -coord_cross.square() / (2.0 * coord_lengthscale.square()))
                else:
                    scaled_cross = (3.0 ** 0.5) * coord_cross / coord_lengthscale
                    kstar = kstar + coord_sigma2 * (1.0 + scaled_cross) * torch.exp(-scaled_cross)
            chol = torch.linalg.cholesky(knn)
            alpha = torch.cholesky_solve(rn.unsqueeze(-1), chol).squeeze(-1)
            solved = torch.cholesky_solve(kstar.unsqueeze(-1), chol).squeeze(-1)
            means.append((kstar * alpha).sum(-1))
            latent_var = (
                (sigma2 if self.representation_kernel else 0.0)
                + (coord_sigma2 if self.base_kernel != "none" and coords_train is not None else 0.0)
                - (kstar * solved).sum(-1))
            variances.append((latent_var + noise).clamp_min(1e-8))
        return torch.cat(means), torch.cat(variances)
