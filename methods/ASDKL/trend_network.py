import torch
import torch.nn as nn
#####Deep Trend Estimation

class TrendNetwork(nn.Module):

    def __init__(

            self,
            hidden_dim,
            width=128,
            depth=3,
            dropout=0.0,
            residual=True):

        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least one")
        self.residual = residual
        self.input_projection = nn.Sequential(nn.Linear(hidden_dim, width), nn.GELU())
        if residual:
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(width, width), nn.GELU(),
                              nn.Dropout(dropout) if dropout > 0 else nn.Identity())
                for _ in range(depth)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
            self.residual_scales = nn.Parameter(torch.full((depth,), 0.1))
        else:
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(width, width), nn.GELU()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.blocks = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def forward(

            self,

            z):

        h = self.input_projection(z)
        if self.residual:
            for scale, block, norm in zip(self.residual_scales, self.blocks, self.norms):
                h = h + scale * block(norm(h))
        else:
            h = self.blocks(h)
        return self.output(h)
