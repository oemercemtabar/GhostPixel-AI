from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import Callback

from models.lightning_module import StegoLightningModule


class BackboneFineTuningCallback(Callback):
    def __init__(self, unfreeze_at_epoch: int, backbone_lr: float) -> None:
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.backbone_lr = backbone_lr
        self._has_unfrozen = False

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: StegoLightningModule) -> None:
        if self._has_unfrozen:
            return
        if trainer.current_epoch < self.unfreeze_at_epoch:
            return
        if not pl_module.hparams.freeze_backbone:
            return

        pl_module.model.unfreeze_backbone()
        optimizer = trainer.optimizers[0]
        optimizer.add_param_group(
            {
                "params": [p for p in pl_module.model.feature_extractor.parameters() if p.requires_grad],
                "lr": self.backbone_lr,
                "weight_decay": pl_module.hparams.weight_decay,
            }
        )
        self._has_unfrozen = True
        pl_module.print(
            f"Unfroze backbone at epoch {trainer.current_epoch} with backbone LR={self.backbone_lr:.2e}"
        )
