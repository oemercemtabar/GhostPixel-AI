from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GHOSTPIXEL_", extra="ignore")

    project_name: str = "GhostPixel-AI"
    app_env: str = "development"
    data_root: Path = Path("data/raw")
    checkpoint_path: Path | None = None
    backbone_name: str = Field(default="efficientnet_v2_s")
    image_size: int = 512
    num_classes: int = 4
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 3e-4


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()

