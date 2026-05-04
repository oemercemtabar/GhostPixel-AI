from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from data.dataset import CLASS_TO_LABEL, GhostPixelDataset, TEST_CLASS_NAME, TEST_LABEL
from data.module import GhostPixelDataModule


def test_dataset_returns_labeled_samples(fake_alaska2_root: Path) -> None:
    dataset = GhostPixelDataset(root_dir=fake_alaska2_root, split="train", balance_strategy="none")

    sample = dataset[0]

    assert sample["class_name"] in CLASS_TO_LABEL
    assert sample["label"] == CLASS_TO_LABEL[sample["class_name"]]
    assert sample["image"].shape[0] == 3


def test_dataset_oversample_balances_training_classes(fake_alaska2_root: Path) -> None:
    dataset = GhostPixelDataset(root_dir=fake_alaska2_root, split="train", balance_strategy="oversample")

    counts = Counter(dataset.samples[idx].label for idx in dataset.indices)

    assert len(set(counts.values())) == 1
    assert set(counts) == set(CLASS_TO_LABEL.values())


def test_dataset_test_split_reads_unlabeled_kaggle_folder(fake_alaska2_root: Path) -> None:
    dataset = GhostPixelDataset(root_dir=fake_alaska2_root, split="test", balance_strategy="none")

    assert len(dataset) == 3
    sample = dataset[1]
    assert sample["class_name"] == TEST_CLASS_NAME
    assert sample["label"] == TEST_LABEL


def test_dataset_raises_when_labeled_folder_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        GhostPixelDataset(root_dir=root, split="train", balance_strategy="none")


def test_data_module_exposes_train_val_and_test_loaders(fake_alaska2_root: Path) -> None:
    datamodule = GhostPixelDataModule(
        data_root=fake_alaska2_root,
        image_size=64,
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup(stage=None)

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))

    assert train_batch["image"].shape[0] == 2
    assert val_batch["image"].shape[0] >= 1
    assert all(label == TEST_LABEL for label in test_batch["label"].tolist())

