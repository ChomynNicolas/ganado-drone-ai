"""Valida etiquetas YOLO de una clase tras la conversión a cattle=0.

No descarga ni convierte datasets. Un directorio se recorre recursivamente,
buscando solo archivos .txt; verificar imágenes y splits corresponde a otra fase.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationResult:
    files: int
    boxes: int
    errors: tuple[str, ...]


def validate_yolo_line(line: str, num_classes: int = 1) -> tuple[int, float, float, float, float]:
    """Verifica una fila de YOLO (class_id x_center y_center width height)."""
    if num_classes <= 0:
        raise ValueError("num_classes debe ser mayor que cero")

    fields = line.split()
    if len(fields) != 5:
        raise ValueError("se esperaban 5 columnas: clase x_center y_center width height")

    if re.fullmatch(r"\d+", fields[0]) is None:
        raise ValueError("class_id debe ser un entero no negativo")
    class_id = int(fields[0])
    if class_id >= num_classes:
        raise ValueError(f"class_id {class_id} fuera de rango [0, {num_classes - 1}]")

    try:
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError as exc:
        raise ValueError("coordenadas no numéricas") from exc

    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("coordenadas no finitas (NaN o infinito)")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("centro fuera del rango normalizado [0, 1]")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("width y height deben estar en (0, 1]")

    tolerance = 1e-6  # tolerar redondeo en cajas que tocan el borde
    if (
        x - width / 2 < -tolerance
        or x + width / 2 > 1.0 + tolerance
        or y - height / 2 < -tolerance
        or y + height / 2 > 1.0 + tolerance
    ):
        raise ValueError("bounding box fuera de los límites de la imagen")

    return class_id, x, y, width, height


def validate_label_file(path: Path, num_classes: int = 1) -> tuple[int, list[str]]:
    """Devuelve número de cajas válidas y errores con archivo/línea."""
    errors: list[str] = []
    boxes = 0
    seen: set[tuple[int, float, float, float, float]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return 0, [f"{path}: no se pudo leer como UTF-8: {exc}"]

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            box = validate_yolo_line(line, num_classes)
        except ValueError as exc:
            errors.append(f"{path}:{number}: {exc}")
            continue
        if box in seen:
            errors.append(f"{path}:{number}: anotación duplicada en la misma imagen")
            continue
        seen.add(box)
        boxes += 1
    return boxes, errors


def validate_labels(path: Path, num_classes: int = 1) -> ValidationResult:
    """Valida un TXT o directorio de TXT, sin modificar los datos de entrada."""
    if num_classes <= 0:
        raise ValueError("num_classes debe ser mayor que cero")
    if not path.exists():
        raise ValueError(f"la ruta no existe: {path}")
    if path.is_file():
        if path.suffix.lower() != ".txt":
            raise ValueError("el archivo de etiquetas debe tener extensión .txt")
        paths = [path]
    elif path.is_dir():
        paths = sorted(item for item in path.rglob("*.txt") if item.is_file())
        if not paths:
            raise ValueError(f"no se encontraron archivos .txt en: {path}")
    else:
        raise ValueError(f"ruta no admitida: {path}")

    total = 0
    errors: list[str] = []
    for label_file in paths:
        boxes, file_errors = validate_label_file(label_file, num_classes=num_classes)
        total += boxes
        errors.extend(file_errors)
    return ValidationResult(files=len(paths), boxes=total, errors=tuple(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validación estática de anotaciones YOLO")
    parser.add_argument("--labels", type=Path, required=True, help="Archivo .txt o directorio")
    parser.add_argument("--num-classes", type=int, default=1, help="Cantidad de clases (1: cattle)")
    args = parser.parse_args(argv)
    try:
        summary = validate_labels(args.labels, args.num_classes)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Archivos revisados: {summary.files}")
    print(f"Cajas válidas: {summary.boxes}")
    print(f"Errores encontrados: {len(summary.errors)}")
    for error in summary.errors:
        print(f"ERROR: {error}")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
