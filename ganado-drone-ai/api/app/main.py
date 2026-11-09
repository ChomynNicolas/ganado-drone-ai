"""FastAPI service for cattle detection on uploaded aerial photographs."""

from __future__ import annotations

import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000


class Detection(BaseModel):
    class_name: str = "cattle"
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    image_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    count: int = Field(ge=0)
    detections: list[Detection]
    inference_ms: float = Field(ge=0)


class Detector:
    """Load one YOLO model per API process and serialize GPU access."""

    def __init__(self, weights: Path, device: str, confidence: float):
        import torch
        from ultralytics import YOLO

        if not weights.is_file():
            raise FileNotFoundError(f"No se encontró el checkpoint configurado: {weights.name}")
        if not 0 < confidence <= 1:
            raise ValueError("MODEL_CONFIDENCE debe estar entre 0 y 1")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("MODEL_DEVICE debe ser auto, cpu o cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE=cuda, pero PyTorch no detecta CUDA")

        self.device = (
            "0" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
        )
        self.checkpoint_name = weights.name
        self.confidence = confidence
        self.model = YOLO(str(weights))
        self.class_id = self._cattle_class_id(self.model.names)
        self._lock = threading.Lock()

    @staticmethod
    def _cattle_class_id(names: dict | list | tuple) -> int:
        items = names.items() if isinstance(names, dict) else enumerate(names)
        matches = [int(index) for index, name in items if str(name).casefold() in {"cattle", "cow"}]
        if len(matches) != 1:
            raise ValueError("El checkpoint debe exponer exactamente una clase cattle o cow")
        return matches[0]

    def predict(self, image: Image.Image, image_name: str) -> PredictionResponse:
        width, height = image.size
        started = time.perf_counter()
        with self._lock:
            result = self.model.predict(
                source=image,
                conf=self.confidence,
                classes=[self.class_id],
                device=self.device,
                verbose=False,
            )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            coords = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for coords_xyxy, confidence, class_id in zip(coords, confidences, classes):
                if int(class_id) == self.class_id:
                    detections.append(
                        Detection(
                            class_name="cattle",
                            confidence=float(confidence),
                            bbox_xyxy=[float(value) for value in coords_xyxy],
                        )
                    )
        return PredictionResponse(
            image_name=image_name,
            width=width,
            height=height,
            count=len(detections),
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
        )


class UnavailableDetector:
    loaded = False
    device = "unavailable"
    checkpoint_name: str | None = None
    confidence: float | None = None

    def __init__(self, error: str):
        self.error = error


def _load_detector() -> Detector | UnavailableDetector:
    weights = Path(os.getenv("MODEL_WEIGHTS", "models/waid-cattle-yolo11n.pt"))
    device = os.getenv("MODEL_DEVICE", "auto").casefold()
    try:
        confidence = float(os.getenv("MODEL_CONFIDENCE", "0.25"))
        return Detector(weights, device, confidence)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        return UnavailableDetector(str(exc))


def create_app(
    detector: Detector | UnavailableDetector | None = None,
    *,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.detector is None:
            app.state.detector = _load_detector()
        yield

    app = FastAPI(
        title="Ganado Drone AI API",
        description="Detección de ganado por imagen; el total de un lote cuenta detecciones, no individuos únicos.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.detector = detector
    app.state.max_image_bytes = max_image_bytes
    app.state.max_image_pixels = max_image_pixels

    @app.get("/health")
    async def health() -> dict[str, str | bool | float | None]:
        active = app.state.detector
        return {
            "status": "ok",
            "model_loaded": bool(active and not isinstance(active, UnavailableDetector)),
            "device": getattr(active, "device", "not_initialized"),
            "checkpoint": getattr(active, "checkpoint_name", None),
            "confidence": getattr(active, "confidence", None),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: Annotated[UploadFile, File(description="Una imagen JPEG o PNG")]):
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or Path(filename).suffix.casefold() not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Solo se admiten archivos JPG, JPEG o PNG")

        content = await file.read(app.state.max_image_bytes + 1)
        await file.close()
        if not content:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        if len(content) > app.state.max_image_bytes:
            raise HTTPException(status_code=413, detail="La imagen excede el límite de 20 MiB")

        try:
            with Image.open(io.BytesIO(content)) as uploaded:
                width, height = uploaded.size
                if width * height > app.state.max_image_pixels:
                    raise HTTPException(
                        status_code=413, detail="La imagen supera el límite de píxeles"
                    )
                uploaded.verify()
            with Image.open(io.BytesIO(content)) as uploaded:
                image = uploaded.convert("RGB")
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=415, detail="El contenido no es una imagen válida"
            ) from exc

        active = app.state.detector
        if active is None:
            raise HTTPException(status_code=503, detail="El detector todavía no está inicializado")
        if isinstance(active, UnavailableDetector):
            raise HTTPException(
                status_code=503, detail=f"El detector no está disponible: {active.error}"
            )

        try:
            return await run_in_threadpool(active.predict, image, filename)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Falló la inferencia de la imagen") from exc

    return app


app = create_app()
