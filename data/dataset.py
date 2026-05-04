from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_TO_LABEL = {
    "Cover": 0,
    "JMiPOD": 1,
    "JUNIWARD": 2,
    "UERD": 3,
}

LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
TEST_CLASS_NAME = "Test"
TEST_LABEL = -1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    class_name: str
    label: int


class GhostPixelDataset(Dataset[dict[str, torch.Tensor | int | str]]):
    """ALASKA2 dataset wrapper for labeled training splits and unlabeled Kaggle test images."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        seed: int = 42,
        train_ratio: float = 0.8,
        balance_strategy: str = "oversample",
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.seed = seed
        self.train_ratio = train_ratio
        self.balance_strategy = balance_strategy
        self.class_names = tuple(class_names or CLASS_TO_LABEL.keys())

        if self.split not in {"train", "val", "all", "test"}:
            raise ValueError("split must be one of: train, val, all, test")
        if self.balance_strategy not in {"none", "oversample", "undersample"}:
            raise ValueError("balance_strategy must be one of: none, oversample, undersample")

        self.samples = self._build_samples()
        self.indices = self._build_indices()

        if not self.indices:
            raise RuntimeError(f"No samples found for split={self.split!r} under {self.root_dir}")

    def _build_samples(self) -> list[SampleRecord]:
        if self.split == "test":
            return self._build_test_samples()

        grouped: dict[str, list[Path]] = defaultdict(list)

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing ALASKA2 class directory: {class_dir}")

            image_paths = sorted(
                path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
            )
            grouped[class_name].extend(image_paths)

        split_samples: list[SampleRecord] = []
        rng = random.Random(self.seed)

        for class_name in self.class_names:
            class_paths = grouped[class_name][:]
            rng.shuffle(class_paths)

            if self.split == "all":
                selected = class_paths
            else:
                cutoff = int(len(class_paths) * self.train_ratio)
                if cutoff <= 0 or cutoff >= len(class_paths):
                    raise ValueError(
                        f"Invalid train_ratio={self.train_ratio} for class {class_name} with {len(class_paths)} files"
                    )
                selected = class_paths[:cutoff] if self.split == "train" else class_paths[cutoff:]

            split_samples.extend(
                SampleRecord(image_path=path, class_name=class_name, label=CLASS_TO_LABEL[class_name])
                for path in selected
            )

        return split_samples

    def _build_test_samples(self) -> list[SampleRecord]:
        test_dir = self.root_dir / TEST_CLASS_NAME
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing ALASKA2 test directory: {test_dir}")

        image_paths = sorted(path for path in test_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        return [
            SampleRecord(image_path=path, class_name=TEST_CLASS_NAME, label=TEST_LABEL)
            for path in image_paths
        ]

    def _build_indices(self) -> list[int]:
        if self.split == "test":
            return list(range(len(self.samples)))

        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            class_to_indices[sample.label].append(idx)

        if self.balance_strategy == "none" or self.split not in {"train", "all"}:
            merged = [idx for indices in class_to_indices.values() for idx in indices]
            return sorted(merged)

        rng = random.Random(self.seed)
        counts = [len(indices) for indices in class_to_indices.values()]
        target_count = max(counts) if self.balance_strategy == "oversample" else min(counts)

        balanced: list[int] = []
        for label in sorted(class_to_indices):
            indices = class_to_indices[label]
            if len(indices) == target_count:
                balanced.extend(indices)
                continue
            if self.balance_strategy == "oversample":
                balanced.extend(indices)
                balanced.extend(rng.choices(indices, k=target_count - len(indices)))
            else:
                balanced.extend(rng.sample(indices, k=target_count))

        rng.shuffle(balanced)
        return balanced

    def _read_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.samples[self.indices[index]]
        image = self._read_image(sample.image_path)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image_tensor,
            "label": sample.label,
            "class_name": sample.class_name,
            "image_path": str(sample.image_path),
        }
