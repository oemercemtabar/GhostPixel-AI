from __future__ import annotations

from pathlib import Path
import sys

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import GhostPixelDataModule
from models import BackboneFineTuningCallback, StegoLightningModule
from settings import get_settings


def main() -> None:
    settings = get_settings()

    datamodule = GhostPixelDataModule(
        data_root=settings.data_root,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    datamodule.setup(stage="fit")
    class_weights = settings.class_weights
    if class_weights is None and settings.use_class_weights:
        class_weights = datamodule.get_train_class_weights(power=settings.class_weight_power)

    model = StegoLightningModule(
        num_classes=settings.num_classes,
        backbone_name=settings.backbone_name,
        learning_rate=settings.learning_rate,
        pretrained=settings.pretrained_backbone,
        freeze_backbone=settings.freeze_backbone,
        scheduler_t_max=settings.scheduler_t_max,
        loss_name=settings.loss_name,
        label_smoothing=settings.label_smoothing,
        focal_gamma=settings.focal_gamma,
        class_weights=class_weights,
    )

    if torch.cuda.is_available():
        precision = "16-mixed"
    else:
        precision = "32-true"

    callbacks: list = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="stegonet-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=settings.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if settings.staged_finetuning and settings.freeze_backbone:
        callbacks.append(
            BackboneFineTuningCallback(
                unfreeze_at_epoch=settings.unfreeze_backbone_epoch,
                backbone_lr=settings.backbone_finetune_learning_rate,
            )
        )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=settings.max_epochs,
        precision=precision,
        accumulate_grad_batches=settings.accumulate_grad_batches,
        limit_train_batches=settings.train_batches_per_epoch,
        limit_val_batches=settings.val_batches_per_epoch,
        log_every_n_steps=25,
        logger=CSVLogger(save_dir="logs", name="train"),
        callbacks=callbacks,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
