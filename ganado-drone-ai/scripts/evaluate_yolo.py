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


def mean_absolute_error(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Las cantidades reales y predichas deben tener el mismo largo no nulo")
    return sum(abs(actual - predicted) for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def count_mae_by_threshold(
    model,
    dataset_root: Path,
    split: str,
    thresholds: list[float],
    *,
    imgsz: int,
    device: int | str,
) -> dict[str, float]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No hay imágenes en {images_dir}")

    ground_truth: list[int] = []
    score_lists: list[list[float]] = []
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
    for result in results:
        stem = Path(result.path).stem
        seen.add(stem)
        ground_truth.append(label_count(labels_dir / f"{stem}.txt"))
        boxes = result.boxes
        if boxes is None:
            score_lists.append([])
        else:
            classes = boxes.cls.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            score_lists.append(
                [float(score) for cls, score in zip(classes, scores) if int(cls) == 0]
            )

    expected = {path.stem for path in image_paths}
    if seen != expected:
        missing = sorted(expected - seen)[:5]
        extra = sorted(seen - expected)[:5]
        raise RuntimeError(
            f"Predicciones no coinciden con imágenes. faltan={missing}, extras={extra}"
        )

    return {
        str(threshold): float(
            mean_absolute_error(
                ground_truth,
                [sum(score >= threshold for score in scores) for scores in score_lists],
            )
        )
        for threshold in thresholds
    }


def test_count_mae(
    model,
    dataset_root: Path,
    threshold: float,
    *,
    imgsz: int,
    device: int | str,
) -> tuple[float, int]:
    images_dir = dataset_root / "images" / "test"
    labels_dir = dataset_root / "labels" / "test"
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
    return float(mean_absolute_error(y_true, y_pred)), len(y_true)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa un YOLO cattle y calibra conteo en valid")
    parser.add_argument("--weights", type=Path, default=Path("models/waid-cattle-yolo11n.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/waid-cattle-clean-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/evaluation.json"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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

        model = YOLO(str(args.weights.resolve()))
        valid_mae = count_mae_by_threshold(
            model,
            data_root,
            "valid",
            thresholds,
            imgsz=args.imgsz,
            device=device,
        )
        selected_threshold = min(
            thresholds, key=lambda threshold: (valid_mae[str(threshold)], -threshold)
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
        test_mae, test_image_count = test_count_mae(
            model,
            data_root,
            selected_threshold,
            imgsz=args.imgsz,
            device=device,
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
                "mae_per_threshold": valid_mae,
                "selected_threshold": selected_threshold,
            },
            "test": {
                "split": "test",
                "images": test_image_count,
                "count_mae_at_validation_threshold": test_mae,
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
        print(f"Umbral seleccionado en valid: {selected_threshold}")
        print(f"MAE de conteo en test: {test_mae:.4f} ({test_image_count} imágenes)")
        for key, value in report["test"]["detector_metrics"].items():
            print(f"{key}: {value:.6f}")
        print(f"Reporte: {args.output.resolve()}")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
