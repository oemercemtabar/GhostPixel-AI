from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt).pow(self.gamma)
        return (focal_term * ce_loss).mean()

