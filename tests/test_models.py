from __future__ import annotations

import pytest
import torch

from models.residual import ForensicResidualLayer
from models.stegonet import StegoNet


def test_forensic_residual_layer_preserves_shape() -> None:
    layer = ForensicResidualLayer(channels=3)
    inputs = torch.randn(2, 3, 64, 64)

    outputs = layer(inputs)

    assert outputs.shape == inputs.shape


@pytest.mark.parametrize("backbone_name", ["efficientnet_v2_s", "swin_t"])
def test_stegonet_forward_shape(backbone_name: str) -> None:
    model = StegoNet(num_classes=4, backbone_name=backbone_name, pretrained=False)
    inputs = torch.randn(1, 3, 224, 224)

    outputs = model(inputs)

    assert outputs.shape == (1, 4)

