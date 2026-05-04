from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GHOSTPIXEL_", extra="ignore")

    project_name: str = "GhostPixel-AI"
    app_env: str = "development"
    data_root: Path = Path("data/raw")
    checkpoint_path: Path | None = None
    backbone_name: str = Field(default="mobilenet_v3_small")
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    image_size: int = 224
    num_classes: int = 4
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 3e-4
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    class_weight_power: float = 1.0
    class_weights: list[float] | None = None
    accumulate_grad_batches: int = 2
    max_epochs: int = 30
    train_batches_per_epoch: int = 2000
    val_batches_per_epoch: int = 200
    scheduler_t_max: int = 30
    staged_finetuning: bool = False
    unfreeze_backbone_epoch: int = 6
    backbone_finetune_learning_rate: float = 5e-5
    early_stopping_patience: int = 10

    @field_validator("class_weights", mode="before")
    @classmethod
    def parse_class_weights(cls, value: object) -> object:
        if value in (None, "", []):
            return None
        if isinstance(value, str):
            return [float(item.strip()) for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
