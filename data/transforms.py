from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_range=(92, 100), compression_type="jpeg", p=0.35),
            A.GaussNoise(std_range=(0.001, 0.01), mean_range=(0.0, 0.0), p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            ToTensorV2(),
        ]
    )

