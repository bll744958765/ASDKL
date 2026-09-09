import torch
import torch.nn as nn
import numpy as np


class AdaptiveFourierEncoder(nn.Module):

    """
    Stage I
    Adaptive Spatial Spectral Encoding
    c_i
      ↓
    ω_i=g(c_i)
      ↓
    γ_i
    """

    def __init__(
            self,
            context_dim,
            aux_dim,
            freq_dim=32):

        super().__init__()

        self.freq_dim = freq_dim

        self.freq_generator = nn.Sequential(

            nn.Linear(
                context_dim + aux_dim,
                128
            ),

            nn.SiLU(),

            nn.Linear(
                128,
                freq_dim * 2
            )
        )

    def generate_frequency(
            self,
            context,x):

        omega = self.freq_generator(
             torch.cat(
                [context, x],
                dim=-1
        ))

        omega = omega.view(
            -1,
            self.freq_dim,
            2
        )

        return omega

    def forward(
            self,
            coordinates,
            context,x):

        """
        coordinates:[B,2]
        context:[B,C]
        """

        omega = self.generate_frequency(
            context,x
        )

        s = coordinates.unsqueeze(1)

        projection = (
            omega * s
        ).sum(-1)

        projection = (
            2.0
            * np.pi
            * projection
        )

        gamma = torch.cat(

            [
                torch.sin(
                    projection
                ),

                torch.cos(
                    projection
                )
            ],

            dim=-1
        )

        gamma = (
            gamma /
            np.sqrt(
                self.freq_dim
            )
        )

        return gamma, omega