from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models.stegonet import StegoNet


class StegoLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "efficientnet_v2_s",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = StegoNet(num_classes=num_classes, backbone_name=backbone_name)
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=images.size(0))

        if stage == "val":
            preds = torch.argmax(logits, dim=1)
            self.val_accuracy.update(preds, labels)
            self.val_f1.update(preds, labels)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

