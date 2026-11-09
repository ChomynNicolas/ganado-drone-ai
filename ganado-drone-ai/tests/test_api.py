from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import Detection, PredictionResponse, UnavailableDetector, create_app


class FakeDetector:
    device = "cpu"
    checkpoint_name = "fake-cattle.pt"
    confidence = 0.4

    def predict(self, image: Image.Image, image_name: str) -> PredictionResponse:
        return PredictionResponse(
            image_name=image_name,
            width=image.width,
            height=image.height,
            count=1,
            detections=[
                Detection(
                    class_name="cattle",
                    confidence=0.9,
                    bbox_xyxy=[1, 2, image.width - 1, image.height - 1],
                )
            ],
            inference_ms=1.25,
        )


def png_bytes(width: int = 16, height: int = 12) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_health_reports_loaded_model() -> None:
    with TestClient(create_app(FakeDetector())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "device": "cpu",
        "checkpoint": "fake-cattle.pt",
        "confidence": 0.4,
    }


def test_predict_returns_boxes_and_count_for_valid_image() -> None:
    with TestClient(create_app(FakeDetector())) as client:
        response = client.post(
            "/predict", files={"file": ("paddock.png", png_bytes(), "image/png")}
        )

    assert response.status_code == 200
    result = response.json()
    assert result["image_name"] == "paddock.png"
    assert result["width"] == 16
    assert result["height"] == 12
    assert result["count"] == len(result["detections"]) == 1
    assert result["detections"][0]["class_name"] == "cattle"


def test_predict_rejects_unsupported_extension() -> None:
    with TestClient(create_app(FakeDetector())) as client:
        response = client.post(
            "/predict", files={"file": ("paddock.gif", png_bytes(), "image/gif")}
        )

    assert response.status_code == 415


def test_predict_rejects_invalid_image_content() -> None:
    with TestClient(create_app(FakeDetector())) as client:
        response = client.post(
            "/predict", files={"file": ("broken.jpg", b"not an image", "image/jpeg")}
        )

    assert response.status_code == 415


def test_predict_enforces_upload_size_limit() -> None:
    with TestClient(create_app(FakeDetector(), max_image_bytes=8)) as client:
        response = client.post("/predict", files={"file": ("large.png", png_bytes(), "image/png")})

    assert response.status_code == 413


def test_predict_reports_unavailable_model() -> None:
    app = create_app(UnavailableDetector("checkpoint ausente"))
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/predict", files={"file": ("paddock.png", png_bytes(), "image/png")}
        )

    assert health.json()["model_loaded"] is False
    assert health.json()["checkpoint"] is None
    assert health.json()["confidence"] is None
    assert response.status_code == 503
