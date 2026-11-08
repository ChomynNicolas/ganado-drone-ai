"""Evaluate cattle detectors against point annotations in the NZ aerial dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CONFIDENCE = 0.40
IMAGE_SIZE = 640
IMAGE_SUFFIX = ".png"
ANNOTATION_SUFFIX = ".png.mask.0.txt"


def read_points(annotation_path: Path, width: int, height: int) -> list[list[float]]:
    points = []
    for line_number, raw_line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Formato de punto inválido en {annotation_path.name}:{line_number}")
        x, y = map(float, parts)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Punto fuera de límites en {annotation_path.name}:{line_number}: {(x, y)}"
            )
        points.append([x, y])
    return points


def load_point_dataset(root: Path, *, allow_partial: bool = False) -> list[dict[str, Any]]:
    images = sorted(
        (
            path
            for path in root.glob("*.png")
            if path.is_file() and not path.name.endswith(".mask.0.png")
        ),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise ValueError(f"No hay imágenes PNG en {root}")

    annotation_paths = {path.name for path in root.glob("*.png.mask.0.txt") if path.is_file()}
    expected_names = {f"{image.name}.mask.0.txt" for image in images}
    if annotation_paths != expected_names:
        raise ValueError(
            "Debe haber una anotación por imagen; "
            f"faltan={sorted(expected_names - annotation_paths)[:5]}, "
            f"sobran={sorted(annotation_paths - expected_names)[:5]}"
        )

    samples = []
    for image_path in images:
        annotation_path = root / f"{image_path.name}.mask.0.txt"
        with Image.open(image_path) as source:
            width, height = source.size
            source.verify()
        points = read_points(annotation_path, width, height)
        samples.append(
            {"image_path": image_path, "points": points, "width": width, "height": height}
        )

    image_count = len(samples)
    point_count = sum(len(sample["points"]) for sample in samples)
    if not allow_partial and (image_count != 655 or point_count != 29_803):
        raise ValueError(
            "Conjunto NZ-Cattle incompleto: se esperaban 655 imágenes y 29.803 puntos; "
            f"se encontraron {image_count} imágenes y {point_count} puntos. "
            "Para una prueba técnica parcial, use --allow-partial."
        )
    return samples


def match_points(
    points: list[list[float]], detections: list[dict[str, Any]], confidence: float = CONFIDENCE
) -> dict[str, int]:
    unmatched = set(range(len(points)))
    true_positives = false_positives = 0
    for detection in sorted(
        (item for item in detections if item["confidence"] >= confidence),
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        candidates = [
            index
            for index in unmatched
            if x1 <= points[index][0] <= x2 and y1 <= points[index][1] <= y2
        ]
        if candidates:
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            matched = min(
                candidates,
                key=lambda index: (
                    (points[index][0] - center_x) ** 2 + (points[index][1] - center_y) ** 2
                ),
            )
            unmatched.remove(matched)
            true_positives += 1
        else:
            false_positives += 1
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": len(unmatched),
    }


def preview(
    sample: dict[str, Any],
    detections: list[dict[str, Any]],
    output_path: Path,
    *,
    confidence: float,
) -> None:
    with Image.open(sample["image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for x, y in sample["points"]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="#24d600", width=1)
    for detection in detections:
        if detection["confidence"] < confidence:
            continue
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline="#ff2424", width=2)
        draw.text(
            (x1, max(0, y1 - 12)), f"{detection['confidence']:.2f}", fill="#ff2424", font=font
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def evaluate_checkpoint(
    samples: list[dict[str, Any]],
    weights: Path,
    output_dir: Path,
    *,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names.items() if isinstance(model.names, dict) else enumerate(model.names)
    class_ids = [int(index) for index, name in names if str(name).casefold() in {"cattle", "cow"}]
    if len(class_ids) != 1:
        raise ValueError(f"El checkpoint debe exponer una clase cattle/cow; nombres={model.names}")

    with Image.open(samples[0]["image_path"]) as source:
        warmup_image = source.convert("RGB")
    model.predict(
        source=warmup_image,
        conf=confidence,
        classes=class_ids,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    predictions = []
    latencies = []
    for sample in samples:
        with Image.open(sample["image_path"]) as source:
            image = source.convert("RGB")
        result = model.predict(
            source=image,
            conf=confidence,
            classes=class_ids,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )[0]
        boxes = result.boxes
        coords = boxes.xyxy.cpu().tolist() if boxes is not None else []
        scores = boxes.conf.cpu().tolist() if boxes is not None else []
        detections = [
            {"bbox_xyxy": [float(value) for value in box], "confidence": float(score)}
            for box, score in zip(coords, scores)
        ]
        predictions.append(detections)
        latencies.append(float(result.speed.get("inference", 0.0)))

    per_image = []
    for sample, detections in zip(samples, predictions):
        matches = match_points(sample["points"], detections, confidence=confidence)
        actual = len(sample["points"])
        predicted = len(detections)
        per_image.append(
            {
                "imagen": sample["image_path"].name,
                "puntos_reales": actual,
                "detecciones": predicted,
                "error_absoluto_conteo": abs(actual - predicted),
                **matches,
            }
        )

    total_tp = sum(row["tp"] for row in per_image)
    total_fp = sum(row["fp"] for row in per_image)
    total_fn = sum(row["fn"] for row in per_image)
    total_points = sum(row["puntos_reales"] for row in per_image)
    total_predicted = sum(row["detecciones"] for row in per_image)
    positive_hits = sum(row["detecciones"] > 0 for row in per_image)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / total_points if total_points else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_mae = mean(row["error_absoluto_conteo"] for row in per_image)

    ranked = sorted(range(len(samples)), key=lambda index: len(samples[index]["points"]))
    preview_indices = sorted({ranked[0], ranked[len(ranked) // 2], ranked[-1]})
    for index in preview_indices:
        preview(
            samples[index],
            predictions[index],
            output_dir / "visualizaciones" / weights.stem / samples[index]["image_path"].name,
            confidence=confidence,
        )

    return {
        "checkpoint": weights.name,
        "device": "cpu",
        "imgsz": IMAGE_SIZE,
        "confidence": confidence,
        "warmup_runs": 1,
        "images": len(samples),
        "ground_truth_points": total_points,
        "detections": total_predicted,
        "tp_point_inside_box": total_tp,
        "fp_boxes_without_point": total_fp,
        "fn_points_without_box": total_fn,
        "precision_point_match": precision,
        "recall_point_match": recall,
        "f1_point_match": f1,
        "mae_count_per_image": count_mae,
        "mean_signed_count_bias": mean(
            row["detecciones"] - row["puntos_reales"] for row in per_image
        ),
        "positive_image_recall": positive_hits / len(per_image) if per_image else 0.0,
        "mean_inference_ms": mean(latencies),
        "per_image": per_image,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evalúa detección y conteo con anotaciones por punto; no calcula mAP de cajas."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Carpeta cow_images")
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=[Path("models/waid-cattle-yolo11n.pt"), Path("models/waid-cattle-yolo11n-5000.pt")],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/nz-cattle-point-evaluation")
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    args = parser.parse_args(argv)

    try:
        samples = load_point_dataset(args.dataset.resolve(), allow_partial=args.allow_partial)
        if not 0 < args.confidence <= 1:
            raise ValueError("La confianza debe estar entre 0 y 1")
        for weights in args.weights:
            if not weights.is_file():
                raise FileNotFoundError(f"No existe el checkpoint: {weights}")
        args.output.mkdir(parents=True, exist_ok=True)
        results = [
            evaluate_checkpoint(samples, weights, args.output, confidence=args.confidence)
            for weights in args.weights
        ]
        complete = (
            len(samples) == 655 and sum(len(sample["points"]) for sample in samples) == 29_803
        )
        report = {
            "dataset": "Remote sensing dataset for detecting cows from high resolution aerial images (Zenodo 5908869)",
            "status": "complete" if complete else "partial_exploratory_sample",
            "annotation_type": "points; not bounding boxes",
            "metrics_note": "No mAP or IoU box metrics are reported. A TP is a predicted box containing one unmatched reference point.",
            "images": len(samples),
            "points": sum(len(sample["points"]) for sample in samples),
            "results": results,
        }
        (args.output / "evaluation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        lines = [
            "# Evaluación externa NZ-Cattle (anotaciones por punto)",
            "",
            f"Estado: **{report['status']}** — {report['images']} imágenes, {report['points']:,} puntos.",
            "",
            f"Protocolo: PIL RGB, CPU, imgsz=640, confianza={args.confidence:.2f}; igual que la API. "
            "Un acierto es una caja que contiene un punto de referencia aún no emparejado.",
            "Al ser etiquetas puntuales, no se reportan mAP ni IoU de cajas.",
            "",
        ]
        for result in results:
            lines.extend(
                [
                    f"## {result['checkpoint']}",
                    "",
                    f"- Precisión / recall / F1 por punto: {result['precision_point_match']:.4f} / "
                    f"{result['recall_point_match']:.4f} / {result['f1_point_match']:.4f}.",
                    f"- MAE de conteo: {result['mae_count_per_image']:.3f} detecciones/imagen; "
                    f"sesgo medio: {result['mean_signed_count_bias']:.3f}.",
                    f"- Recall de imagen positiva: {result['positive_image_recall']:.4f}; "
                    f"latencia media tras calentamiento: {result['mean_inference_ms']:.2f} ms/imagen.",
                    f"- TP={result['tp_point_inside_box']}, FP={result['fp_boxes_without_point']}, "
                    f"FN={result['fn_points_without_box']}.",
                    "",
                ]
            )
        (args.output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for result in results:
            print(
                f"{result['checkpoint']}: imgs={result['images']} puntos={result['ground_truth_points']} "
                f"P/R/F1={result['precision_point_match']:.4f}/{result['recall_point_match']:.4f}/"
                f"{result['f1_point_match']:.4f}, MAE={result['mae_count_per_image']:.3f}"
            )
        print(f"Informe local: {args.output.resolve() / 'evaluation.md'} ({report['status']})")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
