"""Select a count threshold on validation and evaluate the final model on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def label_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Falta etiqueta YOLO para {path.stem}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.split() and line.split()[0] == "0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset_root(dataset_yaml: Path) -> Path:
    for line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "path":
            return Path(json.loads(value.strip()))
    raise ValueError(f"No se encontró una ruta path válida en {dataset_yaml}")


def read_dataset_split(dataset_yaml: Path, root: Path, split: str) -> Path:
    for line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.partition(":")
        if separator and key.strip() == split:
            value = raw_value.strip()
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            path = Path(value)
            return path if path.is_absolute() else root / path
    raise ValueError(f"No se encontró la ruta {split} en {dataset_yaml}")


def labels_dir_for_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    image_indexes = [index for index, part in enumerate(parts) if part.casefold() == "images"]
    if not image_indexes:
        raise ValueError(f"La ruta de imágenes debe contener una carpeta images: {images_dir}")
    parts[image_indexes[-1]] = "labels"
    return Path(*parts)


def mean_absolute_error(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Las cantidades reales y predichas deben tener el mismo largo no nulo")
    return sum(abs(actual - predicted) for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def _read_yolo_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Etiqueta YOLO inválida: {label_path}:{number}")
        class_id, x, y, box_width, box_height = map(float, fields)
        if int(class_id) != 0:
            continue
        boxes.append(
            [
                (x - box_width / 2) * width,
                (y - box_height / 2) * height,
                (x + box_width / 2) * width,
                (y + box_height / 2) * height,
            ]
        )
    return boxes


def _iou(box_a: list[float], box_b: list[float]) -> float:
    intersection = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    )
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _match_boxes(
    targets: list[list[float]],
    predictions: list[dict[str, object]],
    *,
    threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    unmatched = set(range(len(targets)))
    true_positives = false_positives = 0
    selected = sorted(
        (item for item in predictions if float(item["confidence"]) >= threshold),
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    for prediction in selected:
        box = prediction["bbox_xyxy"]
        assert isinstance(box, list)
        candidates = sorted(
            ((_iou(box, targets[index]), index) for index in unmatched), reverse=True
        )
        if candidates and candidates[0][0] >= iou_threshold:
            unmatched.remove(candidates[0][1])
            true_positives += 1
        else:
            false_positives += 1
    return true_positives, false_positives, len(unmatched)


def select_threshold(
    metrics: dict[str, dict[str, float | int]],
    thresholds: list[float],
    *,
    minimum_precision: float = 0.90,
) -> float:
    eligible = [
        threshold
        for threshold in thresholds
        if metrics[str(threshold)]["box_precision_iou_0_50"] >= minimum_precision
    ]
    if not eligible:
        raise ValueError(f"Ningún umbral alcanza precisión {minimum_precision:.0%} en valid")
    return max(
        eligible,
        key=lambda threshold: (
            metrics[str(threshold)]["box_recall_iou_0_50"],
            -metrics[str(threshold)]["count_mae"],
            threshold,
        ),
    )


def count_metrics_by_threshold(
    model,
    dataset_root: Path,
    split: str,
    thresholds: list[float],
    *,
    imgsz: int,
    device: int | str,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
) -> dict[str, dict[str, float | int]]:
    from PIL import Image

    images_dir = images_dir or dataset_root / "images" / split
    labels_dir = labels_dir or dataset_root / "labels" / split
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No hay imágenes en {images_dir}")

    samples: list[dict[str, object]] = []
    results = model.predict(
        source=str(images_dir),
        conf=min(thresholds),
        imgsz=imgsz,
        device=device,
        classes=[0],
        stream=True,
        verbose=False,
    )
    seen: set[str] = set()
    image_by_stem = {path.stem: path for path in image_paths}
    for result in results:
        stem = Path(result.path).stem
        seen.add(stem)
        image_path = image_by_stem.get(stem)
        if image_path is None:
            raise RuntimeError(f"No se reconoce la imagen predicha: {stem}")
        with Image.open(image_path) as image:
            width, height = image.size
        targets = _read_yolo_boxes(labels_dir / f"{stem}.txt", width, height)
        boxes = result.boxes
        if boxes is None:
            detections = []
        else:
            coordinates = boxes.xyxy.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            detections = [
                {
                    "bbox_xyxy": [float(value) for value in coordinates[index]],
                    "confidence": float(scores[index]),
                }
                for index, class_id in enumerate(classes)
                if int(class_id) == 0
            ]
        samples.append({"targets": targets, "detections": detections})

    expected = {path.stem for path in image_paths}
    if seen != expected:
        missing = sorted(expected - seen)[:5]
        extra = sorted(seen - expected)[:5]
        raise RuntimeError(
            f"Predicciones no coinciden con imágenes. faltan={missing}, extras={extra}"
        )

    summaries: dict[str, dict[str, float | int]] = {}
    ground_truth = [len(sample["targets"]) for sample in samples]
    positives = [index for index, count in enumerate(ground_truth) if count > 0]
    negatives = [index for index, count in enumerate(ground_truth) if count == 0]
    for threshold in thresholds:
        predicted = [
            sum(float(item["confidence"]) >= threshold for item in sample["detections"])
            for sample in samples
        ]
        matches = [
            _match_boxes(sample["targets"], sample["detections"], threshold=threshold)
            for sample in samples
        ]
        tp = sum(item[0] for item in matches)
        fp = sum(item[1] for item in matches)
        fn = sum(item[2] for item in matches)
        summaries[str(threshold)] = {
            "count_mae": float(mean_absolute_error(ground_truth, predicted)),
            "box_precision_iou_0_50": tp / (tp + fp) if tp + fp else 0.0,
            "box_recall_iou_0_50": tp / (tp + fn) if tp + fn else 0.0,
            "box_f1_iou_0_50": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "positive_image_recall": (
                sum(predicted[index] > 0 for index in positives) / len(positives)
                if positives
                else 0.0
            ),
            "negative_image_false_positive_rate": (
                sum(predicted[index] > 0 for index in negatives) / len(negatives)
                if negatives
                else 0.0
            ),
            "false_detections_per_negative_image": (
                sum(predicted[index] for index in negatives) / len(negatives) if negatives else 0.0
            ),
            "positive_images": len(positives),
            "negative_images": len(negatives),
        }
    return summaries


def test_count_metrics(
    model,
    dataset_root: Path,
    threshold: float,
    *,
    imgsz: int,
    device: int | str,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
) -> dict[str, float | int]:
    images_dir = images_dir or dataset_root / "images" / "test"
    labels_dir = labels_dir or dataset_root / "labels" / "test"
    y_true: list[int] = []
    y_pred: list[int] = []
    results = model.predict(
        source=str(images_dir),
        conf=threshold,
        imgsz=imgsz,
        device=device,
        classes=[0],
        stream=True,
        verbose=False,
    )
    for result in results:
        stem = Path(result.path).stem
        y_true.append(label_count(labels_dir / f"{stem}.txt"))
        y_pred.append(int(len(result.boxes)) if result.boxes is not None else 0)
    if not y_true:
        raise ValueError("No se generaron predicciones para el conjunto test")
    positives = [index for index, count in enumerate(y_true) if count > 0]
    negatives = [index for index, count in enumerate(y_true) if count == 0]
    return {
        "images": len(y_true),
        "count_mae_at_validation_threshold": float(mean_absolute_error(y_true, y_pred)),
        "positive_image_recall_at_validation_threshold": (
            sum(y_pred[index] > 0 for index in positives) / len(positives) if positives else 0.0
        ),
        "negative_image_false_positive_rate_at_validation_threshold": (
            sum(y_pred[index] > 0 for index in negatives) / len(negatives) if negatives else 0.0
        ),
        "false_detections_per_negative_image_at_validation_threshold": (
            sum(y_pred[index] for index in negatives) / len(negatives) if negatives else 0.0
        ),
        "positive_images": len(positives),
        "negative_images": len(negatives),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa un YOLO cattle y calibra conteo en valid")
    parser.add_argument("--weights", type=Path, default=Path("models/waid-cattle-yolo11n.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/waid-cattle-clean-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/evaluation.json"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    args = parser.parse_args(argv)

    try:
        if not args.weights.is_file():
            raise FileNotFoundError(f"No existe el checkpoint: {args.weights}")
        dataset_root = args.dataset.resolve()
        dataset_yaml = dataset_root / "dataset.yaml"
        data_root = read_dataset_root(dataset_yaml)
        valid_images_dir = read_dataset_split(dataset_yaml, data_root, "val")
        valid_labels_dir = labels_dir_for_images(valid_images_dir)
        test_images_dir = read_dataset_split(dataset_yaml, data_root, "test")
        test_labels_dir = labels_dir_for_images(test_images_dir)

        import torch
        import ultralytics
        from ultralytics import YOLO

        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Se solicitó CUDA, pero PyTorch no detecta una GPU compatible")
        device: int | str = (
            0
            if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        thresholds = sorted(set(args.thresholds))
        if not thresholds or thresholds[0] <= 0 or thresholds[-1] > 1:
            raise ValueError("Los umbrales deben estar en el intervalo (0, 1]")
        if not 0 < args.minimum_precision <= 1:
            raise ValueError("minimum-precision debe estar entre 0 y 1")

        model = YOLO(str(args.weights.resolve()))
        valid_metrics = count_metrics_by_threshold(
            model,
            data_root,
            "valid",
            thresholds,
            imgsz=args.imgsz,
            device=device,
            images_dir=valid_images_dir,
            labels_dir=valid_labels_dir,
        )
        selected_threshold = select_threshold(
            valid_metrics, thresholds, minimum_precision=args.minimum_precision
        )

        test_metrics = model.val(
            data=str(dataset_yaml.resolve()),
            split="test",
            imgsz=args.imgsz,
            device=device,
            conf=0.001,
            iou=0.7,
            plots=True,
            save_json=True,
            project=str((args.output.parent / "ultralytics").resolve()),
            name="waid-test",
            exist_ok=True,
            verbose=True,
        )
        test_count = test_count_metrics(
            model,
            data_root,
            selected_threshold,
            imgsz=args.imgsz,
            device=device,
            images_dir=test_images_dir,
            labels_dir=test_labels_dir,
        )
        test_count_at_040 = test_count_metrics(
            model,
            data_root,
            0.40,
            imgsz=args.imgsz,
            device=device,
            images_dir=test_images_dir,
            labels_dir=test_labels_dir,
        )

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.weights.resolve()),
            "checkpoint_sha256": sha256(args.weights),
            "dataset": str(dataset_yaml.resolve()),
            "dataset_manifest": str((dataset_root / "manifest.json").resolve()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device == 0 else None,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "imgsz": args.imgsz,
            "confidence_threshold_selection": {
                "split": "valid",
                "rule": "maximum recall at IoU 0.50 with box precision at least the configured minimum",
                "minimum_precision": args.minimum_precision,
                "metrics_per_threshold": valid_metrics,
                "mae_per_threshold": {
                    threshold: values["count_mae"] for threshold, values in valid_metrics.items()
                },
                "selected_threshold": selected_threshold,
            },
            "test": {
                "split": "test",
                **test_count,
                "count_metrics_at_confidence_0_40": test_count_at_040,
                "detector_metrics": {
                    key: float(value)
                    for key, value in test_metrics.results_dict.items()
                    if isinstance(value, (int, float))
                },
                "ultralytics_save_dir": str(test_metrics.save_dir),
            },
            "interpretation": (
                "Conteo por imagen; no representa individuos únicos entre fotos superpuestas."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"Umbral seleccionado en valid (precision >= {args.minimum_precision:.0%}): "
            f"{selected_threshold}"
        )
        print(
            "MAE de conteo en test: "
            f"{test_count['count_mae_at_validation_threshold']:.4f} "
            f"({test_count['images']} imágenes)"
        )
        print(
            "FPR por imagen negativa en test: "
            f"{test_count['negative_image_false_positive_rate_at_validation_threshold']:.4f}"
        )
        for key, value in report["test"]["detector_metrics"].items():
            print(f"{key}: {value:.6f}")
        print(f"Reporte: {args.output.resolve()}")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
