"""Fine-tune YOLO11n on the locally prepared WAID cattle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_device(preference: str, cuda_available: bool) -> int | str:
    if preference == "auto":
        return 0 if cuda_available else "cpu"
    if preference == "cuda" and not cuda_available:
        raise ValueError("Se solicitó CUDA, pero PyTorch no detecta una GPU compatible")
    return 0 if preference == "cuda" else "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrenar YOLO11n con WAID cattle")
    parser.add_argument("--config", type=Path, default=Path("configs/train_waid_cattle.toml"))
    parser.add_argument("--epochs", type=int, help="Sobrescribe epochs del YAML")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)

    try:
        config = tomllib.loads(args.config.read_text(encoding="utf-8"))
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.device is not None:
            config["device"] = args.device

        dataset = Path(config["dataset"]).resolve()
        initial_weights = Path(config["initial_weights"]).resolve()
        output_weights = Path(config["output_weights"]).resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"No existe dataset.yaml: {dataset}")
        if not initial_weights.is_file():
            raise FileNotFoundError(f"No existe checkpoint inicial: {initial_weights}")
        if output_weights.exists():
            raise FileExistsError(f"El checkpoint destino ya existe: {output_weights}")
        if int(config["epochs"]) < 1:
            raise ValueError("epochs debe ser mayor que cero")

        import torch
        import ultralytics
        from ultralytics import YOLO

        device = select_device(str(config["device"]), torch.cuda.is_available())
        run_project = Path(config["run_project"]).resolve()
        run_name = str(config["run_name"])
        run_dir = run_project / run_name
        if run_dir.exists():
            raise FileExistsError(f"El directorio de entrenamiento ya existe: {run_dir}")

        model = YOLO(str(initial_weights))
        model.train(
            data=str(dataset),
            epochs=int(config["epochs"]),
            imgsz=int(config["imgsz"]),
            batch=int(config["batch"]),
            patience=int(config["patience"]),
            seed=int(config["seed"]),
            workers=int(config["workers"]),
            device=device,
            project=str(run_project),
            name=run_name,
            exist_ok=False,
            deterministic=True,
            plots=True,
            save=True,
            val=True,
            verbose=True,
        )

        best_weights = Path(model.trainer.best)
        if not best_weights.is_file():
            raise RuntimeError(f"El entrenamiento no produjo best.pt en {best_weights}")
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, output_weights)
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_yaml": str(dataset),
            "dataset_manifest": str(dataset.parent / "manifest.json"),
            "initial_weights": str(initial_weights),
            "initial_weights_sha256": sha256(initial_weights),
            "best_weights": str(best_weights.resolve()),
            "output_weights": str(output_weights),
            "output_weights_sha256": sha256(output_weights),
            "run_dir": str(run_dir),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device == 0 else None,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "parameters": {
                key: config[key]
                for key in (
                    "epochs",
                    "imgsz",
                    "batch",
                    "patience",
                    "seed",
                    "workers",
                )
            },
        }
        metadata_path = output_weights.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Best checkpoint: {output_weights}")
        print(f"SHA-256: {metadata['output_weights_sha256']}")
        print(f"Metadatos: {metadata_path}")
        print(f"Run: {run_dir}")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
