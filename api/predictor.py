from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
import torch

from data.dataset import LABEL_TO_CLASS
from data.transforms import build_eval_transforms
from models import StegoLightningModule
from settings import AppSettings


class GhostPixelPredictor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_eval_transforms(settings.image_size)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        checkpoint_path = self.settings.checkpoint_path
        if checkpoint_path and checkpoint_path.exists():
            module = StegoLightningModule.load_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                num_classes=self.settings.num_classes,
                backbone_name=self.settings.backbone_name,
                learning_rate=self.settings.learning_rate,
            )
            return module.model

        module = StegoLightningModule(
            num_classes=self.settings.num_classes,
            backbone_name=self.settings.backbone_name,
            learning_rate=self.settings.learning_rate,
        )
        return module.model

    def predict_bytes(self, payload: bytes) -> dict[str, float | str | None]:
        image = Image.open(BytesIO(payload)).convert("RGB")
        image_np = np.asarray(image)
        tensor = self.transform(image=image_np)["image"].unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        label = int(prediction.item())
        return {
            "class_name": LABEL_TO_CLASS[label],
            "confidence_score": float(confidence.item()),
            "explainability_map": None,
        }
