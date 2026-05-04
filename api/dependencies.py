from functools import lru_cache

from api.predictor import GhostPixelPredictor
from settings import get_settings


@lru_cache(maxsize=1)
def get_predictor() -> GhostPixelPredictor:
    return GhostPixelPredictor(settings=get_settings())

