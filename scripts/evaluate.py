from __future__ import annotations

import lightning as L

from data import GhostPixelDataModule
from models import StegoLightningModule
from settings import get_settings


def main() -> None:
    settings = get_settings()
    if settings.checkpoint_path is None:
        raise ValueError("Set GHOSTPIXEL_CHECKPOINT_PATH to a trained Lightning checkpoint before evaluation.")

    datamodule = GhostPixelDataModule(
        data_root=settings.data_root,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        balance_strategy="none",
    )
    model = StegoLightningModule.load_from_checkpoint(
        checkpoint_path=str(settings.checkpoint_path),
        num_classes=settings.num_classes,
        backbone_name=settings.backbone_name,
        learning_rate=settings.learning_rate,
    )

    trainer = L.Trainer(accelerator="auto", devices=1)
    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()

