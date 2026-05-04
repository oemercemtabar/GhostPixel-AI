from __future__ import annotations

import torch
from torch import nn


class ForensicResidualLayer(nn.Module):
    """SRM-inspired high-pass preprocessing with a learnable fusion scale."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()

        kernel = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 2.0, -4.0, 2.0, 0.0],
                [-1.0, -4.0, 12.0, -4.0, -1.0],
                [0.0, 2.0, -4.0, 2.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ) / 12.0

        weight = kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
        self.filter = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.filter.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.filter(x)
        return x + (self.scale * residual)

