"""Audita la estructura y las etiquetas originales del dataset WAID.

Ejecutar desde la raíz del proyecto:
    python -m scripts.audit_waid --dataset data/raw/waid-repo/WAID

Solo lee imágenes y etiquetas. No transforma datasets ni mueve archivos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.validate_yolo_labels import validate_yolo_line

WAID_CLASSES = ("sheep", "cattle", "seal", "camelus", "kiang", "zebra")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SPLITS = ("train", "valid", "test")


@dataclass
class SplitSummary:
    images: int = 0
    label_files: int = 0
    valid_boxes: int = 0
    cattle_boxes: int = 0
    cattle_images: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    classes: dict[str, int] = field(default_factory=dict)


@dataclass
class AuditResult:
    dataset: str
    source_classes: list[str]
    source_cattle_id: int
    splits: dict[str, SplitSummary]
    errors: list[str]
    warnings: list[str]
    hash_check: bool

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_boxes(path: Path) -> tuple[Counter, list[str]]:
    """Cuenta anotaciones válidas por clase; no cuenta duplicados exactos."""
    counters: Counter = Counter()
    errors: list[str] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return counters, [f"{path}: lectura fallida: {exc}"]

    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            box = validate_yolo_line(raw, num_classes=len(WAID_CLASSES))
        except ValueError as exc:
            errors.append(f"{path}:{number}: {exc}")
            continue
        if box in seen:
            errors.append(f"{path}:{number}: anotación exacta duplicada")
            continue
        seen.add(box)
        counters[WAID_CLASSES[box[0]]] += 1
    return counters, errors


def audit_dataset(dataset: Path, hash_check: bool = False) -> AuditResult:
    root = dataset.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    splits: dict[str, SplitSummary] = {}

    classes_file = root / "classes.txt"
    if not classes_file.is_file():
        errors.append(f"No existe el archivo obligatorio: {classes_file}")
    else:
        try:
            actual = tuple(
                line.strip()
                for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            )
        except (OSError, UnicodeError) as exc:
            actual = ()
            errors.append(f"No se pudo leer classes.txt: {exc}")
        if actual != WAID_CLASSES:
            errors.append(
                f"Mapeo de clases inesperado: {list(actual)}; esperado {list(WAID_CLASSES)}. "
                "No continuar con la conversión hasta revisar la fuente."
            )

    hashes: dict[str, tuple[str, str]] = {}
    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        info = SplitSummary(classes={name: 0 for name in WAID_CLASSES})
        splits[split] = info

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            errors.append(f"Split {split}: faltan imágenes o etiquetas en {img_dir} / {lbl_dir}")
            continue

        images: dict[str, Path] = {}
        for path in sorted(img_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.stem in images:
                errors.append(f"Split {split}: varias imágenes con el mismo stem: {path.stem}")
            else:
                images[path.stem] = path

        labels = {path.stem: path for path in lbl_dir.glob("*.txt") if path.is_file()}
        info.images = len(images)
        info.label_files = len(labels)

        if not images:
            errors.append(f"Split {split}: no hay imágenes compatibles")
        for stem, image_path in images.items():
            label = labels.get(stem)
            if label is None:
                info.missing_labels += 1
            else:
                counts, label_errors = read_boxes(label)
                errors.extend(label_errors)
                info.valid_boxes += sum(counts.values())
                info.cattle_boxes += counts["cattle"]
                info.cattle_images += int(counts["cattle"] > 0)
                for class_name in WAID_CLASSES:
                    info.classes[class_name] += counts[class_name]

            if hash_check:
                try:
                    digest = sha256_file(image_path)
                except OSError as exc:
                    errors.append(f"No se pudo leer imagen {image_path}: {exc}")
                    continue
                previous = hashes.get(digest)
                if previous and previous[0] != split:
                    errors.append(
                        f"Imagen idéntica en splits distintos: "
                        f"{previous[0]}/{previous[1]} y {split}/{image_path.name}"
                    )
                hashes.setdefault(digest, (split, image_path.name))

        info.orphan_labels = len(set(labels) - set(images))
        if info.orphan_labels:
            errors.append(f"Split {split}: {info.orphan_labels} etiquetas sin imagen asociada")
        if info.missing_labels:
            warnings.append(
                f"Split {split}: {info.missing_labels} imágenes sin TXT asociado; "
                "confirmar si son negativos legítimos"
            )
        if info.cattle_images == 0:
            warnings.append(f"Split {split}: ninguna imagen etiquetada como cattle")

    if not hash_check:
        warnings.append(
            "No se calcularon hashes de imágenes: no se ha comprobado duplicación "
            "exacta entre splits. Usar --hash-check."
        )
    warnings.append(
        "Los hashes no detectan imágenes casi idénticas ni escenas repetidas. "
        "Se requiere inspección visual y revisar procedencia por vuelo."
    )
    warnings.append(
        "La licencia de la fuente oficial WAID no está confirmada. "
        "No redistribuir imágenes hasta revisarla."
    )
    return AuditResult(
        dataset=str(root),
        source_classes=list(WAID_CLASSES),
        source_cattle_id=1,
        splits=splits,
        errors=errors,
        warnings=warnings,
        hash_check=hash_check,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoría del dataset original WAID")
    parser.add_argument("--dataset", type=Path, required=True, help="Directorio con classes.txt")
    parser.add_argument("--output", type=Path, default=None, help="Informe JSON local opcional")
    parser.add_argument(
        "--hash-check", action="store_true", help="Detecta imágenes idénticas entre splits"
    )
    args = parser.parse_args(argv)
    result = audit_dataset(args.dataset, hash_check=args.hash_check)

    print("WAID: auditoría de estructura y anotaciones")
    for split, info in result.splits.items():
        print(
            f"{split}: images={info.images} labels={info.label_files} "
            f"cattle_images={info.cattle_images} cattle_boxes={info.cattle_boxes} "
            f"missing_labels={info.missing_labels}"
        )
    print(f"Errores: {len(result.errors)}; advertencias: {len(result.warnings)}")
    for issue in result.errors:
        print(f"ERROR: {issue}")
    for warning in result.warnings:
        print(f"AVISO: {warning}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Informe guardado en: {args.output}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
