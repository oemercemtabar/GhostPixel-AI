from __future__ import annotations

import lightning as L
import torch
from torch import nn

from data.dataset import LABEL_TO_CLASS
from models.losses import FocalLoss
from models.stegonet import StegoNet


class StegoLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "mobilenet_v3_small",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        scheduler_t_max: int = 24,
        loss_name: str = "cross_entropy",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = StegoNet(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.num_classes = num_classes
        self.class_names = [LABEL_TO_CLASS[idx] for idx in range(num_classes)]
        self.register_buffer("val_confusion_matrix", torch.zeros(num_classes, num_classes, dtype=torch.float32))
        class_weights_tensor = None
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        if loss_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights_tensor,
                label_smoothing=label_smoothing,
            )
        elif loss_name == "focal":
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights_tensor,
                label_smoothing=label_smoothing,
            )
        else:
            raise ValueError("loss_name must be 'cross_entropy' or 'focal'")

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
            self._update_confusion_matrix(preds=preds, labels=labels)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_validation_epoch_start(self) -> None:
        self.val_confusion_matrix.zero_()

    def on_validation_epoch_end(self) -> None:
        matrix = self.val_confusion_matrix
        total = matrix.sum()
        correct = torch.diag(matrix).sum()
        val_acc = correct / total.clamp_min(1.0)

        true_positives = torch.diag(matrix)
        false_positives = matrix.sum(dim=0) - true_positives
        false_negatives = matrix.sum(dim=1) - true_positives
        precision = true_positives / (true_positives + false_positives).clamp_min(1.0)
        recall = true_positives / (true_positives + false_negatives).clamp_min(1.0)
        f1_per_class = (2 * precision * recall) / (precision + recall).clamp_min(1e-8)
        val_f1 = f1_per_class.mean()
        prediction_distribution = matrix.sum(dim=0) / total.clamp_min(1.0)
        target_distribution = matrix.sum(dim=1) / total.clamp_min(1.0)

        self.log("val_acc", val_acc, prog_bar=True, sync_dist=False)
        self.log("val_f1", val_f1, prog_bar=True, sync_dist=False)
        self._log_per_class_metrics(precision, recall, f1_per_class, prediction_distribution, target_distribution)
        self._print_validation_report(
            matrix=matrix,
            val_acc=val_acc,
            val_f1=val_f1,
            precision=precision,
            recall=recall,
            f1_per_class=f1_per_class,
            prediction_distribution=prediction_distribution,
            target_distribution=target_distribution,
        )

    def _update_confusion_matrix(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        preds = preds.detach().view(-1).to(torch.long)
        labels = labels.detach().view(-1).to(torch.long)
        valid_mask = (labels >= 0) & (labels < self.num_classes)
        preds = preds[valid_mask]
        labels = labels[valid_mask]

        if preds.numel() == 0:
            return

        encoded = labels * self.num_classes + preds
        counts = torch.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.val_confusion_matrix += counts.view(self.num_classes, self.num_classes).to(
            self.val_confusion_matrix.device,
            dtype=self.val_confusion_matrix.dtype,
        )

    def _log_per_class_metrics(
        self,
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1_per_class: torch.Tensor,
        prediction_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> None:
        for class_idx, class_name in enumerate(self.class_names):
            metric_prefix = class_name.lower()
            self.log(f"val_precision_{metric_prefix}", precision[class_idx], sync_dist=False)
            self.log(f"val_recall_{metric_prefix}", recall[class_idx], sync_dist=False)
            self.log(f"val_f1_{metric_prefix}", f1_per_class[class_idx], sync_dist=False)
            self.log(f"val_pred_share_{metric_prefix}", prediction_distribution[class_idx], sync_dist=False)
            self.log(f"val_target_share_{metric_prefix}", target_distribution[class_idx], sync_dist=False)

    def _print_validation_report(
        self,
        matrix: torch.Tensor,
        val_acc: torch.Tensor,
        val_f1: torch.Tensor,
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1_per_class: torch.Tensor,
        prediction_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> None:
        report_lines = [
            "",
            "Validation report",
            f"overall: acc={val_acc.item():.4f} macro_f1={val_f1.item():.4f}",
        ]

        for class_idx, class_name in enumerate(self.class_names):
            report_lines.append(
                (
                    f"{class_name:<8} "
                    f"precision={precision[class_idx].item():.4f} "
                    f"recall={recall[class_idx].item():.4f} "
                    f"f1={f1_per_class[class_idx].item():.4f} "
                    f"target_share={target_distribution[class_idx].item():.4f} "
                    f"pred_share={prediction_distribution[class_idx].item():.4f}"
                )
            )

        report_lines.append("confusion_matrix:")
        report_lines.extend(
            "  " + " ".join(f"{int(value):6d}" for value in row.tolist())
            for row in matrix.to(torch.int64)
        )
        self.print("\n".join(report_lines))

    def configure_optimizers(self) -> dict[str, object]:
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.hparams.scheduler_t_max)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
