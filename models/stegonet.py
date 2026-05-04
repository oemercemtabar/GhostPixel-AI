from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Small_Weights,
    Swin_T_Weights,
    efficientnet_b0,
    efficientnet_v2_s,
    mobilenet_v3_small,
    swin_t,
)

from models.residual import ForensicResidualLayer


class StegoNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.residual_layer = ForensicResidualLayer(channels=3)
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[3].in_features
            backbone.classifier = nn.Sequential(
                nn.Linear(backbone.classifier[0].in_features, backbone.classifier[0].out_features),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            feature_module = backbone.features
        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            feature_module = backbone.features
        elif backbone_name == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_v2_s(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            feature_module = backbone.features
        elif backbone_name == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = swin_t(weights=weights)
            in_features = backbone.head.in_features
            backbone.head = nn.Linear(in_features, num_classes)
            feature_module = backbone.features
        else:
            raise ValueError(
                "backbone_name must be 'mobilenet_v3_small', 'efficientnet_b0', "
                "'efficientnet_v2_s' or 'swin_t'"
            )

        self.feature_extractor = feature_module
        if freeze_backbone:
            self.freeze_backbone()

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_layer(x)
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = True
