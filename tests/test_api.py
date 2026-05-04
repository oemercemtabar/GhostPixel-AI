from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.dependencies import get_predictor
from api.main import app


class StubPredictor:
    def predict_bytes(self, payload: bytes) -> dict[str, float | str | None]:
        assert payload
        return {
            "class_name": "Cover",
            "confidence_score": 0.91,
            "explainability_map": None,
        }


def _make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_healthcheck() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_renders_web_console() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "GhostPixel-AI" in response.text
    assert "Run Detection" in response.text


def test_detect_rejects_non_image_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/detect",
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image."


def test_detect_returns_prediction_payload() -> None:
    app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    client = TestClient(app)

    response = client.post(
        "/detect",
        files={"file": ("sample.jpg", _make_test_image_bytes(), "image/jpeg")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "class_name": "Cover",
        "confidence_score": 0.91,
        "explainability_map": None,
    }
