"""Genera un mosaico local de anotaciones de un dataset YOLO cattle=0."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

from scripts.audit_waid import IMAGE_SUFFIXES
from scripts.validate_yolo_labels import validate_yolo_line


def create_preview(
    dataset: Path,
    output: Path,
    *,
    split: str = "train",
    count: int = 9,
    columns: int = 3,
    seed: int = 42,
) -> int:
    if split not in ("train", "valid", "test"):
        raise ValueError("split inválido")
    if count <= 0 or columns <= 0:
        raise ValueError("count y columns deben ser positivos")

    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("No se encontraron imágenes y etiquetas del split elegido")
    files = [
        file
        for file in image_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise ValueError("No hay imágenes para visualizar")
    files.sort(
        key=lambda path: (
            hashlib.sha256(f"{seed}:{split}:{path.name}".encode()).digest(),
            path.name,
        )
    )
    files = files[:count]

    tile_w, tile_h = 360, 390
    rows = math.ceil(len(files) / columns)
    montage = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    painter = ImageDraw.Draw(montage)

    for index, image_file in enumerate(files):
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.is_file():
            raise ValueError(f"Falta etiqueta: {label_file}")
        x0 = (index % columns) * tile_w
        y0 = (index // columns) * tile_h
        try:
            with Image.open(image_file) as source_image:
                image = source_image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Imagen ilegible {image_file}: {exc}") from exc
        image.thumbnail((tile_w - 12, tile_h - 40))
        actual_w, actual_h = image.size
        x_img = x0 + (tile_w - actual_w) // 2
        y_img = y0 + 28
        montage.paste(image, (x_img, y_img))
        painter.text((x0 + 8, y0 + 6), image_file.name[:44], fill="black")
        boxes = 0
        for number, raw in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                _, xc, yc, w, h = validate_yolo_line(raw, num_classes=1)
            except ValueError as exc:
                raise ValueError(f"{label_file}:{number}: {exc}") from exc
            x_left = x_img + (xc - w / 2) * actual_w
            y_top = y_img + (yc - h / 2) * actual_h
            x_right = x_img + (xc + w / 2) * actual_w
            y_bottom = y_img + (yc + h / 2) * actual_h
            painter.rectangle((x_left, y_top, x_right, y_bottom), outline="red", width=2)
            boxes += 1
        painter.text((x0 + 8, y0 + tile_h - 18), f"cattle: {boxes}", fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output, quality=92)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspección visual de cajas cattle")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/cattle-preview.jpg"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        total = create_preview(
            args.dataset,
            args.output,
            split=args.split,
            count=args.count,
            columns=args.columns,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Mosaico: {total} imágenes → {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
