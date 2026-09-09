
# models/asdkl.py

import torch
import torch.nn as nn

try:
    from .context_encoder import ContextEncoder
    from .adaptive_fourier_encoder import AdaptiveFourierEncoder
    from .spectral_attention import SpectralAttention
    from .trend_network import TrendNetwork
except ImportError:  # Direct execution from methods/ASDKL.
    from context_encoder import ContextEncoder
    from adaptive_fourier_encoder import AdaptiveFourierEncoder
    from spectral_attention import SpectralAttention
    from trend_network import TrendNetwork


class ASDKL(nn.Module):

    """
    Adaptive Spectral Deep Kernel Learning
    """

    def __init__(
            self,
            coord_dim,
            aux_dim,
            context_dim=32,
            freq_dim=32,
            hidden_dim=64,
            ablation="full",
            fusion="gated",
            trend_width=128,
            trend_depth=3,
            dropout=0.0,
            trend_residual=True,
            local_anchor=True):

        super().__init__()
        self.ablation = ablation
        self.fusion = fusion
        self.local_anchor = local_anchor
        # Fixed, non-trainable random frequencies used only by the
        # ``fixed_fourier`` ablation.  The experiment seed controls this buffer.
        fixed_generator = torch.Generator().manual_seed(314159)
        self.register_buffer("fixed_omega", torch.randn(freq_dim, 2, generator=fixed_generator))

        ################################################
        # Stage I
        ################################################

        self.context_encoder = ContextEncoder(

            coord_dim,
            aux_dim,
            hidden_dim,
            context_dim
        )

        self.fourier_encoder = AdaptiveFourierEncoder(

            context_dim=context_dim,
            aux_dim=aux_dim,
            freq_dim=freq_dim
        )

        # Explicit target--neighbour relation encoding. Coordinates and
        # covariates are standardized using training data before this module.
        self.relation_encoder = nn.Sequential(
            nn.Linear(coord_dim + aux_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ################################################
        # Stage II
        ################################################

        feature_dim = (

            context_dim
            +
            aux_dim
            +
            2 * freq_dim
            +
            hidden_dim
        )

        self.attention = SpectralAttention(

            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            spectral_weight=1.0
        )

        # A gated residual path preserves target-location information after
        # neighborhood aggregation and stabilizes attention-based prediction.
        self.center_projection = nn.Linear(feature_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        ################################################
        # Trend Network
        ################################################

        self.trend_net = TrendNetwork(hidden_dim, width=trend_width,
                                      depth=trend_depth, dropout=dropout,
                                      residual=trend_residual)

        # A local response anchor gives the mean function a conservative
        # geostatistical path.  The gate can retain this interpolation in
        # locally smooth regions and favour the neural trend where attributes
        # or spectral structure are more informative.
        self.anchor_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.log_anchor_temperature = nn.Parameter(torch.tensor(-1.0))

    def forward(

            self,

            s,
            x,

            neigh_s,
            neigh_x,
            neigh_y):

        """
        Parameters
        ----------

        s :
            [B,2]

        x :
            [B,P]

        neigh_s :
            [B,K,2]

        neigh_x :
            [B,K,P]

        neigh_y :
            [B,K,1]
        """

        ################################################
        # Context Encoding
        ################################################

        context = self.context_encoder(

            neigh_s,
            neigh_x,
            neigh_y
        )
        if self.ablation == "no_context":
            context = torch.zeros_like(context)

        ################################################
        # Adaptive Fourier
        ################################################

        gamma_center, omega = (

            self.fourier_encoder(

                coordinates=s,
                context=context,
                x=x
            )
        )
        if self.ablation == "fixed_fourier":
            omega = self.fixed_omega.unsqueeze(0).expand(s.shape[0], -1, -1)
            projection = 2.0 * torch.pi * (omega * s.unsqueeze(1)).sum(-1)
            gamma_center = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1) / (omega.shape[1] ** 0.5)
        if self.ablation == "no_fourier":
            gamma_center, omega = torch.zeros_like(gamma_center), torch.zeros_like(omega)

        ################################################
        # Neighbor Context
        ################################################

        B, K, _ = neigh_s.shape

        context_neighbor = (

            context
            .unsqueeze(1)
            .repeat(1, K, 1)
        )

        ################################################
        # Neighbor Fourier
        ################################################

        if self.ablation == "independent_frequencies":
            gamma_neighbor, _ = self.fourier_encoder(
                coordinates=neigh_s.reshape(-1, 2),
                context=context_neighbor.reshape(-1, context.shape[-1]),
                x=neigh_x.reshape(-1, neigh_x.shape[-1]),
            )
            gamma_neighbor = gamma_neighbor.view(B, K, -1)
        else:
            # The target-conditioned frequency system Omega_i is shared by
            # the target and every candidate neighbour, yielding a common
            # local spectral chart.
            projection_neighbor = 2.0 * torch.pi * torch.einsum(
                "bmd,bkd->bkm", omega, neigh_s)
            gamma_neighbor = torch.cat(
                [torch.sin(projection_neighbor),
                 torch.cos(projection_neighbor)],
                dim=-1,
            ) / (omega.shape[1] ** 0.5)
        if self.ablation == "no_fourier":
            gamma_neighbor = torch.zeros_like(gamma_neighbor)

        ################################################
        # Explicit target--neighbour relation
        ################################################

        relative_s = neigh_s - s.unsqueeze(1)
        relative_x = neigh_x - x.unsqueeze(1)
        relative_distance = torch.linalg.vector_norm(
            relative_s, dim=-1, keepdim=True)
        relation_neighbor = self.relation_encoder(torch.cat(
            [relative_s, relative_x, relative_distance], dim=-1))
        relation_center = self.relation_encoder(torch.zeros(
            (B, relative_s.shape[-1] + relative_x.shape[-1] + 1),
            dtype=s.dtype, device=s.device))
        if self.ablation == "no_relation_encoder":
            relation_neighbor = torch.zeros_like(relation_neighbor)
            relation_center = torch.zeros_like(relation_center)

        ################################################
        # Node Feature
        #
        # u_i=[c_i,gamma_i,x_i,r_0]
        ################################################

        u_center = torch.cat(

            [
                context,
                gamma_center,
                x,
                relation_center
            ],

            dim=-1
        )

        ################################################
        # Neighbor Feature
        ################################################

        u_neighbor = torch.cat(

            [
                context_neighbor,
                gamma_neighbor,
                neigh_x,
                relation_neighbor
            ],

            dim=-1
        )

        ################################################
        # Spectral Attention
        ################################################

        h, alpha = self.attention(

            u_center=u_center,

            u_neighbors=u_neighbor,

            gamma_center=gamma_center,

            gamma_neighbors=gamma_neighbor
        )
        if self.ablation == "mean_aggregation":
            h = self.attention.W_v(u_neighbor).mean(dim=1)
            alpha = torch.full(alpha.shape, 1.0 / alpha.shape[1], device=alpha.device)

        if self.fusion == "gated":
            center_h = self.center_projection(u_center)
            gate = self.fusion_gate(torch.cat([center_h, h], dim=-1))
            h = self.fusion_norm(gate * h + (1.0 - gate) * center_h)

        ################################################
        # Trend
        ################################################

        neural_mu = self.trend_net(h)
        if self.local_anchor:
            distance = torch.linalg.vector_norm(neigh_s - s.unsqueeze(1), dim=-1)
            temperature = torch.nn.functional.softplus(self.log_anchor_temperature) + 1e-3
            local_weight = torch.softmax(-distance / temperature, dim=1)
            local_mu = (local_weight * neigh_y).sum(dim=1, keepdim=True)
            anchor_gate = self.anchor_gate(h)
            mu = anchor_gate * neural_mu + (1.0 - anchor_gate) * local_mu
        else:
            local_mu = torch.zeros_like(neural_mu)
            anchor_gate = torch.ones_like(neural_mu)
            mu = neural_mu

        ################################################
        # Return
        ################################################

        return {

            "mu": mu,

            "h": h,

            "context": context,

            "gamma": gamma_center,

            "kernel_feature": torch.cat([gamma_center, h], dim=-1),

            "omega": omega,

            "attention": alpha
            ,"neural_mu": neural_mu
            ,"local_mu": local_mu
            ,"anchor_gate": anchor_gate
            ,"relation_neighbor": relation_neighbor
            ,"gamma_neighbor": gamma_neighbor
        }

