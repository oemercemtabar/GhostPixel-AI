from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from data.dataset import GhostPixelDataset
from data.transforms import build_eval_transforms, build_train_transforms


class GhostPixelDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
        balance_strategy: str = "oversample",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_strategy = balance_strategy

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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting val_dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting test_dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
