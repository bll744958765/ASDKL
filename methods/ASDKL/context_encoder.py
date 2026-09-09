import torch
import torch.nn as nn


class ContextEncoder(nn.Module):

    def __init__(

            self,

            coord_dim,

            aux_dim,

            hidden_dim=64,

            context_dim=32):

        super().__init__()

        input_dim = (
            coord_dim
            +
            aux_dim
            +
            1
        )

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                context_dim
            )
        )

    def forward(

            self,

            neigh_s,

            neigh_x,

            neigh_y):

        context = torch.cat(

            [

                neigh_s,

                neigh_x,

                neigh_y if neigh_y.ndim == 3 else neigh_y.unsqueeze(-1)

            ],

            dim=-1

        )

        feat = self.encoder(
            context
        )

        c = feat.mean(
            dim=1
        )

        return c
