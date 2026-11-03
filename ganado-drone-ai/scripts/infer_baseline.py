"""Baseline de inferencia: YOLO11n preentrenado COCO, clase 'cow'.

No es un modelo ajustado a WAID. No reporta métricas mAP ni censos únicos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def image_paths(source: Path, limit: int) -> list[Path]:
    """Selecciona un conjunto pequeño de imágenes para un smoke test."""
    if limit < 1:
        raise ValueError("--limit debe ser mayor que cero")
    if source.is_file():
        images = [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    elif source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )[:limit]
    else:
        raise ValueError(f"No existe el origen de imágenes: {source}")
    if not images:
        raise ValueError("No se encontraron imágenes JPG/PNG")
    return images[:limit]


def cow_class_id(names: dict | list | tuple) -> int:
    """Busca el ID de la clase COCO cow sin asumir reindexación WAID."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    ids = [int(index) for index, label in items if str(label).casefold() == "cow"]
    if len(ids) != 1:
        raise ValueError("El checkpoint no expone exactamente una clase 'cow' (COCO)")
    return ids[0]


def image_size(path: Path) -> tuple[int, int]:
    """Verifica que sea una imagen decodificable, no solo extensión correcta."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"Imagen no válida: {path.name}") from exc


def cow_detections(result, class_id: int) -> list[dict]:
    """Convierte los tensores de Ultralytics a JSON y filtra la clase objetivo."""
    boxes = result.boxes
    if boxes is None:
        return []
    coords = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    if not (len(coords) == len(confidences) == len(classes)):
        raise ValueError("Resultado YOLO inconsistente")
    return [
        {
            "class_name": "cow",
            "confidence": round(float(score), 6),
            "bbox_xyxy": [round(float(value), 2) for value in xyxy],
        }
        for xyxy, score, cls in zip(coords, confidences, classes)
        if int(cls) == class_id
    ]


def select_device(preference: str, cuda_available: bool) -> str:
    if preference == "auto":
        return "0" if cuda_available else "cpu"
    if preference == "cuda" and not cuda_available:
        raise ValueError("Se solicitó GPU CUDA, pero PyTorch no la detecta")
    return "0" if preference == "cuda" else "cpu"


def infer_images(
    model,
    paths: list[Path],
    output_dir: Path,
    *,
    confidence: float,
    image_size_px: int,
    device: str,
    model_sha256: str,
) -> dict:
    """Ejecuta inferencia y guarda imágenes anotadas y resumen sin rutas privadas."""
    if not (0 < confidence <= 1):
        raise ValueError("--conf debe estar entre 0 (exclusivo) y 1")
    if image_size_px < 128:
        raise ValueError("--imgsz debe ser al menos 128")
    class_id = cow_class_id(model.names)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, path in enumerate(paths, start=1):
        width, height = image_size(path)
        started = time.perf_counter()
        results = model.predict(
            source=str(path),
            conf=confidence,
            imgsz=image_size_px,
            classes=[class_id],
            device=device,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"Se esperaba una predicción para {path.name}")
        prediction = results[0]
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        detections = cow_detections(prediction, class_id)
        annotated = output_dir / f"{index:03d}_{path.stem}.jpg"
        prediction.save(filename=str(annotated))
        reported_ms = prediction.speed.get("inference") if prediction.speed else None
        items.append(
            {
                "image": path.name,
                "width": width,
                "height": height,
                "count": len(detections),
                "detections": detections,
                "annotated_image": annotated.name,
                "model_inference_ms": reported_ms,
                "wall_predict_ms": wall_ms,
            }
        )

    report = {
        "model": "yolo11n.pt",
        "weights_sha256": model_sha256,
        "task": "COCO pretrained cow smoke test (not fine-tuned on WAID)",
        "cow_class_id": class_id,
        "device": device,
        "confidence": confidence,
        "imgsz": image_size_px,
        "images_processed": len(items),
        "total_detections": sum(item["count"] for item in items),
        "disclaimer": "Detecciones acumuladas, no cabezas de ganado únicas entre imágenes",
        "results": items,
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inferencia baseline YOLO11n sobre imágenes")
    parser.add_argument("--source", type=Path, required=True, help="Imagen o carpeta de imágenes")
    parser.add_argument("--weights", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/baseline"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        parser.error("Checkpoint ausente. Ejecutar: python -m scripts.download_baseline")
    try:
        paths = image_paths(args.source, args.limit)
        import torch
        from ultralytics import YOLO

        selected = select_device(args.device, torch.cuda.is_available())
        model = YOLO(str(args.weights))
        with args.weights.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        report = infer_images(
            model,
            paths,
            args.output,
            confidence=args.conf,
            image_size_px=args.imgsz,
            device=selected,
            model_sha256=digest,
        )
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Dispositivo: {report['device']}")
    print(f"Imágenes procesadas: {report['images_processed']}")
    print(f"Detecciones cow acumuladas: {report['total_detections']}")
    print(f"Resultados: {args.output.resolve() / 'predictions.json'}")
    print("Aviso: baseline COCO; aún no se entrenó el modelo para fotografías aéreas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
