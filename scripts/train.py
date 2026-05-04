from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from data import GhostPixelDataModule
from models import StegoLightningModule
from settings import get_settings


def main() -> None:
    settings = get_settings()

    datamodule = GhostPixelDataModule(
        data_root=settings.data_root,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    model = StegoLightningModule(
        num_classes=settings.num_classes,
        backbone_name=settings.backbone_name,
        learning_rate=settings.learning_rate,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=20,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=CSVLogger(save_dir="logs", name="train"),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="stegonet-{epoch:02d}-{val_acc:.4f}",
                monitor="val_acc",
                mode="max",
                save_top_k=3,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
