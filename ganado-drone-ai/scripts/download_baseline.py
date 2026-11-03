"""Descarga el checkpoint oficial YOLO11n a models/, sin añadirlo a Git."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def download_checkpoint(target: Path, loader=None) -> str:
    """Obtiene un checkpoint y devuelve su SHA-256.

    `loader` permite probar la lógica sin descargar archivos ni usar PyTorch.
    """
    target = target.resolve()
    if target.name != "yolo11n.pt":
        raise ValueError("Solo se admite el checkpoint oficial yolo11n.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        if loader is None:
            from ultralytics import YOLO

            loader = YOLO
        original_dir = Path.cwd()
        try:
            os.chdir(target.parent)
            loader(target.name)
        finally:
            os.chdir(original_dir)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("No se descargó models/yolo11n.pt; revisar conexión y permisos")
    with target.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descargar checkpoint YOLO11n a models/")
    parser.add_argument("--output", type=Path, default=Path("models/yolo11n.pt"))
    args = parser.parse_args(argv)
    try:
        digest = download_checkpoint(args.output)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Checkpoint: {args.output.resolve()}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
