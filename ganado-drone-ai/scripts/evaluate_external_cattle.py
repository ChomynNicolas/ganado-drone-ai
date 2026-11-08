"""Evaluate cattle checkpoints on an external Supervisely-format UAV set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
OPERATIONAL_CONFIDENCE = 0.40
AP_CONFIDENCE_FLOOR = 0.001
IMAGE_SIZE = 640
IOU_THRESHOLDS = [round(0.50 + index * 0.05, 2) for index in range(10)]


def parse_supervisely_boxes(
    annotation: dict[str, Any], width: int, height: int
) -> list[list[float]]:
    """Read cattle rectangles from Supervisely JSON without rewriting the source file."""
    boxes = []
    for obj in annotation.get("objects", []):
        if obj.get("classTitle") != "cattle":
            continue
        if obj.get("geometryType") != "rectangle":
            raise ValueError("Se encontró una anotación cattle que no es rectangular")
        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) != 2 or any(len(point) != 2 for point in exterior):
            raise ValueError("Caja cattle con coordenadas Supervisely inválidas")
        x1, y1 = map(float, exterior[0])
        x2, y2 = map(float, exterior[1])
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            raise ValueError("Caja cattle con coordenadas no finitas")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Caja cattle fuera de la imagen {width}x{height}: {(x1, y1, x2, y2)}")
        boxes.append([x1, y1, x2, y2])
    return boxes


def load_samples(dataset2_dir: Path, *, allow_partial: bool) -> list[dict[str, Any]]:
    image_dir = dataset2_dir / "img"
    annotation_dir = dataset2_dir / "ann"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Se esperan las carpetas img y ann dentro de {dataset2_dir}")

    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise ValueError(f"No hay imágenes en {image_dir}")

    annotation_files = {path.name for path in annotation_dir.glob("*.json") if path.is_file()}
    samples = []
    for image_path in images:
        annotation_path = annotation_dir / f"{image_path.name}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Falta anotación para {image_path.name}")
        with Image.open(image_path) as source:
            width, height = source.size
            source.verify()
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        recorded_size = annotation.get("size", {})
        if recorded_size and (
            recorded_size.get("width") != width or recorded_size.get("height") != height
        ):
            raise ValueError(f"Dimensiones de anotación no coinciden para {image_path.name}")
        samples.append(
            {
                "image_path": image_path,
                "ground_truth": parse_supervisely_boxes(annotation, width, height),
                "width": width,
                "height": height,
            }
        )

    expected_annotation_names = {f"{image.name}.json" for image in images}
    if annotation_files != expected_annotation_names:
        missing = sorted(expected_annotation_names - annotation_files)
        extra = sorted(annotation_files - expected_annotation_names)
        raise ValueError(
            f"Etiquetas sobrantes o sin imagen: faltan={missing[:5]}, extras={extra[:5]}"
        )

    image_count = len(samples)
    box_count = sum(len(sample["ground_truth"]) for sample in samples)
    if not allow_partial and (image_count != 14 or box_count != 62):
        raise ValueError(
            "Dataset 2 incompleto: se requieren 14 imágenes y 62 cajas; "
            f"se encontraron {image_count} imágenes y {box_count} cajas. "
            "Para una prueba técnica no concluyente, use --allow-partial."
        )
    return samples


def iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_predictions(
    samples: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    *,
    confidence: float,
    iou_threshold: float,
) -> dict[str, int | float]:
    per_image = [
        match_one_image(
            sample["ground_truth"],
            image_predictions,
            confidence=confidence,
            iou_threshold=iou_threshold,
        )
        for sample, image_predictions in zip(samples, predictions)
    ]
    true_positives = sum(int(item["true_positives"]) for item in per_image)
    false_positives = sum(int(item["false_positives"]) for item in per_image)
    false_negatives = sum(int(item["false_negatives"]) for item in per_image)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def match_one_image(
    targets: list[list[float]],
    image_predictions: list[dict[str, Any]],
    *,
    confidence: float,
    iou_threshold: float,
) -> dict[str, int]:
    true_positives = false_positives = 0
    matched_targets: set[int] = set()
    selected = sorted(
        (item for item in image_predictions if item["confidence"] >= confidence),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    for prediction in selected:
        candidates = sorted(
            (
                (iou(prediction["bbox_xyxy"], target), index)
                for index, target in enumerate(targets)
                if index not in matched_targets
            ),
            reverse=True,
        )
        if candidates and candidates[0][0] >= iou_threshold:
            matched_targets.add(candidates[0][1])
            true_positives += 1
        else:
            false_positives += 1
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": len(targets) - len(matched_targets),
    }


def average_precision(
    samples: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    *,
    confidence_floor: float,
    iou_threshold: float,
) -> float:
    from ultralytics.utils.metrics import compute_ap

    total_targets = sum(len(sample["ground_truth"]) for sample in samples)
    if total_targets == 0:
        return 0.0
    detections = [
        (sample_index, item)
        for sample_index, items in enumerate(predictions)
        for item in items
        if item["confidence"] >= confidence_floor
    ]
    detections.sort(key=lambda pair: pair[1]["confidence"], reverse=True)
    matched: list[set[int]] = [set() for _ in samples]
    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    tp = fp = 0
    for sample_index, detection in detections:
        targets = samples[sample_index]["ground_truth"]
        candidates = sorted(
            (
                (iou(detection["bbox_xyxy"], target), index)
                for index, target in enumerate(targets)
                if index not in matched[sample_index]
            ),
            reverse=True,
        )
        if candidates and candidates[0][0] >= iou_threshold:
            matched[sample_index].add(candidates[0][1])
            tp += 1
        else:
            fp += 1
        cumulative_tp.append(tp)
        cumulative_fp.append(fp)
    if not detections:
        return 0.0

    recalls = [value / total_targets for value in cumulative_tp]
    precisions = [t / (t + f) for t, f in zip(cumulative_tp, cumulative_fp)]
    ap, _, _ = compute_ap(recalls, precisions)
    return float(ap)


def count_metrics(
    samples: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    box_metrics_by_image: list[dict[str, int]],
    *,
    confidence: float,
) -> dict[str, Any]:
    actual_counts = [len(sample["ground_truth"]) for sample in samples]
    predicted_counts = [
        sum(item["confidence"] >= confidence for item in items) for items in predictions
    ]
    positive_indices = [index for index, count in enumerate(actual_counts) if count > 0]
    detected_positives = sum(predicted_counts[index] > 0 for index in positive_indices)
    return {
        "mae_por_imagen": mean(
            abs(actual - predicted) for actual, predicted in zip(actual_counts, predicted_counts)
        ),
        "imagenes_positivas": len(positive_indices),
        "recall_de_imagen_positiva": detected_positives / len(positive_indices)
        if positive_indices
        else None,
        "imagenes_negativas": sum(count == 0 for count in actual_counts),
        "falsas_alarmas_por_imagen_negativa": None,
        "detalle_por_imagen": [
            {
                "imagen": sample["image_path"].name,
                "conteo_real": actual,
                "conteo_predicho": predicted,
                "error_absoluto_de_conteo": abs(actual - predicted),
                **box_metrics,
            }
            for sample, actual, predicted, box_metrics in zip(
                samples, actual_counts, predicted_counts, box_metrics_by_image
            )
        ],
    }


def draw_preview(
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
    width = max(3, round(min(image.size) / 500))
    for detection in detections:
        if detection["confidence"] < confidence:
            continue
        box = detection["bbox_xyxy"]
        draw.rectangle(box, outline="#ff3030", width=width)
        draw.text(
            (box[0], max(0, box[1] - 14)),
            f"cattle {detection['confidence']:.2f}",
            fill="#ff3030",
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def evaluate_checkpoint(
    samples: list[dict[str, Any]],
    weights: Path,
    output_dir: Path,
    *,
    confidence: float = OPERATIONAL_CONFIDENCE,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names.items() if isinstance(model.names, dict) else enumerate(model.names)
    class_ids = [int(index) for index, name in names if str(name).casefold() in {"cattle", "cow"}]
    if len(class_ids) != 1:
        raise ValueError(f"El checkpoint debe tener una clase cattle/cow; nombres={model.names}")

    with Image.open(samples[0]["image_path"]) as source:
        warmup_image = source.convert("RGB")
    model.predict(
        source=warmup_image,
        conf=AP_CONFIDENCE_FLOOR,
        classes=class_ids,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    all_predictions = []
    timings = []
    for sample in samples:
        with Image.open(sample["image_path"]) as source:
            image = source.convert("RGB")
        result = model.predict(
            source=image,
            conf=AP_CONFIDENCE_FLOOR,
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
        all_predictions.append(detections)
        timings.append(float(result.speed.get("inference", 0.0)))
        draw_preview(
            sample,
            detections,
            output_dir / "visualizaciones" / weights.stem / sample["image_path"].name,
            confidence=confidence,
        )

    fixed = match_predictions(
        samples,
        all_predictions,
        confidence=confidence,
        iou_threshold=0.50,
    )
    per_image_metrics = [
        match_one_image(
            sample["ground_truth"],
            detections,
            confidence=confidence,
            iou_threshold=0.50,
        )
        for sample, detections in zip(samples, all_predictions)
    ]
    ap_by_iou = {
        f"{threshold:.2f}": average_precision(
            samples,
            all_predictions,
            confidence_floor=AP_CONFIDENCE_FLOOR,
            iou_threshold=threshold,
        )
        for threshold in IOU_THRESHOLDS
    }
    return {
        "checkpoint": weights.name,
        "imgsz": IMAGE_SIZE,
        "confidence_operativa": confidence,
        "confianza_minima_para_ap": AP_CONFIDENCE_FLOOR,
        "dispositivo": "cpu",
        "inference_warmup_runs": 1,
        "cajas_reales": sum(len(sample["ground_truth"]) for sample in samples),
        "cajas_predichas_al_umbral_operativo": sum(
            item["confidence"] >= confidence for items in all_predictions for item in items
        ),
        "precision_recall_f1_iou_0_50_a_0_40": fixed,
        "map_0_50": ap_by_iou["0.50"],
        "map_0_50_0_95": mean(ap_by_iou.values()),
        "ap_por_iou": ap_by_iou,
        "conteo": count_metrics(samples, all_predictions, per_image_metrics, confidence=confidence),
        "latencia_inferencia_ms_promedio": mean(timings),
        "latencias_inferencia_ms": timings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compara checkpoints en Dataset 2 sin modificar ni usar sus datos para entrenar."
    )
    parser.add_argument(
        "--dataset2", type=Path, required=True, help="Carpeta Dataset2 con subcarpetas img/ y ann/"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=[Path("models/waid-cattle-yolo11n.pt"), Path("models/waid-cattle-yolo11n-5000.pt")],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/external-cattle-dataset2")
    )
    parser.add_argument("--confidence", type=float, default=OPERATIONAL_CONFIDENCE)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permite una prueba técnica exploratoria; no satisface la evaluación completa de 14 imágenes/62 cajas.",
    )
    args = parser.parse_args(argv)

    try:
        samples = load_samples(args.dataset2.resolve(), allow_partial=args.allow_partial)
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
            len(samples) == 14 and sum(len(sample["ground_truth"]) for sample in samples) == 62
        )
        report = {
            "estado": "completo" if complete else "muestra_parcial_no_concluyente",
            "fuente": "Cattle Detection and Counting in UAV Images, Dataset 2",
            "licencia_declarada_en_paquete": "CC BY-NC-ND 4.0",
            "imagenes": len(samples),
            "cajas_reales": sum(len(sample["ground_truth"]) for sample in samples),
            "protocolo": (
                f"Evaluación externa a confianza {args.confidence:.2f}; no se entrenó ni calibró "
                "con Dataset 2; conteos por imagen, no individuos únicos."
            ),
            "resultados": results,
        }
        (args.output / "evaluacion.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lines = [
            "# Evaluación externa: Dataset 2 de ganado UAV",
            "",
            f"Estado: **{report['estado']}** — {len(samples)} imágenes y {report['cajas_reales']} cajas anotadas.",
            "",
            f"Protocolo fijo: CPU, resolución 640, confianza de conteo {args.confidence:.2f} e IoU 0,50 para P/R/F1. "
            "La AP usa el ordenamiento de confianza desde 0,001. No se ajustó ningún umbral con este conjunto.",
            "Los conteos son por fotografía; no representan individuos únicos entre imágenes superpuestas.",
            "Los archivos originales y las visualizaciones se mantienen localmente; no redistribuir el conjunto.",
            "",
        ]
        for result in results:
            count = result["conteo"]
            fixed = result["precision_recall_f1_iou_0_50_a_0_40"]
            lines.extend(
                [
                    f"## {result['checkpoint']}",
                    "",
                    f"- Precisión / recall / F1 (IoU 0,50; confianza {args.confidence:.2f}): "
                    f"{fixed['precision']:.4f} / {fixed['recall']:.4f} / {fixed['f1']:.4f}.",
                    f"- mAP@0.5 / mAP@0.5:0.95: {result['map_0_50']:.4f} / {result['map_0_50_0_95']:.4f}.",
                    f"- MAE de conteo: {count['mae_por_imagen']:.4f}; recall de imagen positiva: "
                    f"{count['recall_de_imagen_positiva'] if count['recall_de_imagen_positiva'] is not None else 'N/D'}.",
                    f"- Latencia del modelo: {result['latencia_inferencia_ms_promedio']:.2f} ms/imagen "
                    "(promedio tras un calentamiento).",
                    "",
                ]
            )
            for detail in count["detalle_por_imagen"]:
                lines.append(
                    f"- {detail['imagen']}: real={detail['conteo_real']}, "
                    f"predicho={detail['conteo_predicho']}, TP={detail['true_positives']}, "
                    f"FP={detail['false_positives']}, FN={detail['false_negatives']}."
                )
            lines.append("")
        (args.output / "evaluacion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"Estado: {report['estado']} ({len(samples)} imágenes, {report['cajas_reales']} cajas)"
        )
        for result in results:
            print(
                f"{result['checkpoint']}: conteo MAE={result['conteo']['mae_por_imagen']:.4f}, "
                f"mAP50={result['map_0_50']:.4f}, mAP50-95={result['map_0_50_0_95']:.4f}"
            )
        print(f"Informe local: {args.output.resolve() / 'evaluacion.md'}")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
