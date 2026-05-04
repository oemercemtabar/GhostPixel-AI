from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from data.dataset import GhostPixelDataset
from data.transforms import build_eval_transforms, build_train_transforms


class GhostPixelDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 256,
        batch_size: int = 4,
        num_workers: int = 2,
        balance_strategy: str = "oversample",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_strategy = balance_strategy
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = 2 if self.num_workers > 0 else None

        self.train_dataset: GhostPixelDataset | None = None
        self.val_dataset: GhostPixelDataset | None = None
        self.test_dataset: GhostPixelDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = GhostPixelDataset(
                root_dir=self.data_root,
                split="train",
                transform=build_train_transforms(self.image_size),
                balance_strategy=self.balance_strategy,
            )
            self.val_dataset = GhostPixelDataset(
                root_dir=self.data_root,
                split="val",
                transform=build_eval_transforms(self.image_size),
                balance_strategy="none",
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = GhostPixelDataset(
                root_dir=self.data_root,
                split="test",
                transform=build_eval_transforms(self.image_size),
                balance_strategy="none",
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting train_dataloader")
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting val_dataloader")
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting test_dataloader")
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def _build_dataloader(self, dataset: GhostPixelDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def get_train_class_weights(self, power: float = 1.0) -> list[float]:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before computing class weights")

        counts = torch.zeros(len(self.train_dataset.class_names), dtype=torch.float32)
        for sample in self.train_dataset.samples:
            counts[sample.label] += 1.0

        weights = counts.sum() / counts.clamp_min(1.0)
        weights = weights.pow(power)
        weights = weights / weights.mean().clamp_min(1e-8)
        return weights.tolist()
