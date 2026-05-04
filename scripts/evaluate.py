from __future__ import annotations

from pathlib import Path
import sys

import lightning as L

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    datamodule.setup(stage="fit")
    class_weights = settings.class_weights
    if class_weights is None and settings.use_class_weights:
        class_weights = datamodule.get_train_class_weights(power=settings.class_weight_power)

    model = StegoLightningModule.load_from_checkpoint(
        checkpoint_path=str(settings.checkpoint_path),
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

    trainer = L.Trainer(accelerator="auto", devices=1)
    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
