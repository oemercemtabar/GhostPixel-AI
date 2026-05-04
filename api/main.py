from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from api.dependencies import get_predictor
from api.predictor import GhostPixelPredictor
from api.schemas import DetectionResponse
from settings import get_settings

settings = get_settings()
app = FastAPI(title=settings.project_name, version="0.1.0")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_steganography(
    file: UploadFile = File(...),
    predictor: GhostPixelPredictor = Depends(get_predictor),
) -> DetectionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = predictor.predict_bytes(payload)
    return DetectionResponse(**result)

