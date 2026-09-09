# models/spectral_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    """
    Spectral-aware Neighborhood Interaction Module

    Input:
        u_center      : [B, F]
        u_neighbors   : [B, K, F]

        gamma_center  : [B, G]
        gamma_neighbors : [B, K, G]

    Output:
        h : [B, hidden_dim]
    """

    def __init__(
            self,
            input_dim,
            hidden_dim=128,
            spectral_weight=1.0):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.spectral_weight = spectral_weight

        self.W_q = nn.Linear(
            input_dim,
            hidden_dim,
            bias=False
        )

        self.W_k = nn.Linear(
            input_dim,
            hidden_dim,
            bias=False
        )

        self.W_v = nn.Linear(
            input_dim,
            hidden_dim,
            bias=False
        )

    def forward(
            self,
            u_center,
            u_neighbors,
            gamma_center,
            gamma_neighbors):

        """
        Parameters
        ----------
        u_center :
            [B,F]

        u_neighbors :
            [B,K,F]

        gamma_center :
            [B,G]

        gamma_neighbors :
            [B,K,G]

        Returns
        -------
        h :
            [B,hidden_dim]

        alpha :
            [B,K]
        """

        # --------------------------------------------------
        # Query
        # --------------------------------------------------

        Q = self.W_q(u_center)

        # [B,1,H]
        Q = Q.unsqueeze(1)

        # --------------------------------------------------
        # Key / Value
        # --------------------------------------------------

        K = self.W_k(u_neighbors)

        V = self.W_v(u_neighbors)

        # --------------------------------------------------
        # Feature attention
        #
        # Q_i^T K_j
        # --------------------------------------------------

        feature_score = torch.sum(
            Q * K,
            dim=-1
        ) / (self.hidden_dim ** 0.5)

        # [B,K]

        # --------------------------------------------------
        # Spectral similarity
        #
        # gamma_i^T gamma_j
        # --------------------------------------------------

        gamma_center = gamma_center.unsqueeze(1)

        spectral_score = torch.sum(
            gamma_center * gamma_neighbors,
            dim=-1
        ) / (gamma_neighbors.shape[-1] ** 0.5)

        # [B,K]

        # --------------------------------------------------
        # Final attention
        # --------------------------------------------------

        score = (
            feature_score
            +
            self.spectral_weight
            * spectral_score
        )

        alpha = F.softmax(
            score,
            dim=-1
        )

        # --------------------------------------------------
        # Neighborhood aggregation
        # --------------------------------------------------

        alpha_expand = alpha.unsqueeze(-1)

        h = torch.sum(
            alpha_expand * V,
            dim=1
        )

        return h, alpha
