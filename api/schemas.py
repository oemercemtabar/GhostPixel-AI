from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    class_name: str = Field(description="Predicted steganography class name.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Softmax confidence for the top class.")
    explainability_map: str | None = Field(
        default=None,
        description="Placeholder URI or identifier for a future explainability artifact.",
    )

