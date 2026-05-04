from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.dependencies import get_predictor
from api.predictor import GhostPixelPredictor
from api.schemas import DetectionResponse
from settings import get_settings

settings = get_settings()
app = FastAPI(title=settings.project_name, version="0.1.0")
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))

app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
app.mount("/assets", StaticFiles(directory=str(project_root / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse)
def web_app(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"project_name": settings.project_name},
    )


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
