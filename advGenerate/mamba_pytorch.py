# src/advGenerate/mamba_pytorch.py

import torch
import torch.nn as nn
from mamba_ssm import Mamba as MambaSSM

class DynamicFeatureProjection(nn.Module):
    """
    Replaces static batch averaging with a learned, parametric projection matrix.
    Safely projects variable batch and channel features into the d_model plane.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=True)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projection(x)
        return self.layer_norm(out)


class Mamba(nn.Module):
    """
    A drop-in replacement that uses the official hardware-accelerated mamba-ssm
    while keeping your custom projection and residual structure intact.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()

        # Add the dynamic projection front-end to ingest network features cleanly
        self.feature_projection = DynamicFeatureProjection(d_model, d_model)

        # Official Mamba block (instantiated 2 times to match your previous 2-layer setup)
        self.layers = nn.ModuleList(
            [
                MambaSSM(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(2)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Process via learned dynamic projection
        x = self.feature_projection(x)

        # 2. Sequential forward execution utilizing fast CUDA kernels with residual connections
        for layer in self.layers:
            # mamba-ssm computes the heavily optimized block;
            # we manually add the residual connection to stabilize training
            layer_out = layer(x)
            x = x + layer_out

        # 3. Final layer norm
        return self.norm_f(x)
