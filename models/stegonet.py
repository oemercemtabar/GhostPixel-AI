from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    Swin_T_Weights,
    efficientnet_v2_s,
    swin_t,
)

from models.residual import ForensicResidualLayer


class StegoNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "efficientnet_v2_s",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.residual_layer = ForensicResidualLayer(channels=3)
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_v2_s(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        elif backbone_name == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = swin_t(weights=weights)
            in_features = backbone.head.in_features
            backbone.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("backbone_name must be 'efficientnet_v2_s' or 'swin_t'")

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_layer(x)
        return self.backbone(x)

