from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TEST_CLASS_NAME


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="JPEG")


@pytest.fixture()
def fake_alaska2_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"

    class_counts = {
        "Cover": 4,
        "JMiPOD": 2,
        "JUNIWARD": 3,
        "UERD": 2,
    }

    seed = 0
    for class_name, count in class_counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            _write_image(class_dir / f"{class_name.lower()}_{idx}.jpg", seed=seed)
            seed += 1

    test_dir = root / TEST_CLASS_NAME
    test_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        _write_image(test_dir / f"test_{idx}.jpg", seed=seed)
        seed += 1

    return root
