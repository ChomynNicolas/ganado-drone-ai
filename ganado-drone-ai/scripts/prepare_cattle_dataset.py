"""Convierte WAID (cattle=1) a un dataset YOLO monocategoría (cattle=0).

Conserva train/valid/test sin mezclarlos y se niega a escribir sobre datos
procesados existentes. No modifica los archivos de origen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scripts.audit_waid import IMAGE_SUFFIXES, SPLITS, WAID_CLASSES, audit_dataset
from scripts.validate_yolo_labels import validate_labels, validate_yolo_line

CATTLE_ID = WAID_CLASSES.index("cattle")


@dataclass(frozen=True)
class ImageRecord:
    image: Path
    cattle_lines: tuple[str, ...]

    @property
    def has_cattle(self) -> bool:
        return bool(self.cattle_lines)


def _read_cattle_lines(label_path: Path) -> tuple[str, ...]:
    lines: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, *_ = validate_yolo_line(raw, num_classes=len(WAID_CLASSES))
        if class_id == CATTLE_ID:
            # Mantener las coordenadas originales sin redondear nuevamente.
            coordinates = raw.split()[1:]
            lines.append("0 " + " ".join(coordinates))
    return tuple(lines)


def _stable_sample(
    records: list[ImageRecord], limit: int, seed: int, split: str
) -> list[ImageRecord]:
    """Subconjunto reproducible independiente del orden del sistema de archivos."""
    ranked = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(f"{seed}:{split}:{record.image.name}".encode()).digest(),
            record.image.name,
        ),
    )
    return ranked[:limit]


def _records(source: Path, split: str) -> tuple[list[ImageRecord], int]:
    image_dir = source / "images" / split
    labels_dir = source / "labels" / split
    records: list[ImageRecord] = []
    skipped = 0
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            skipped += 1
            continue  # Nunca interpretar silenciosamente una etiqueta ausente como negativa.
        records.append(ImageRecord(image=image, cattle_lines=_read_cattle_lines(label)))
    return records, skipped


def _check_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"No existe el directorio de origen: {source}")
    if output.exists():
        raise ValueError(f"El destino ya existe; usar un nombre nuevo: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("El origen y el destino no pueden contenerse entre sí")


def prepare_dataset(
    source: Path,
    output: Path,
    *,
    seed: int = 42,
    train_negative_ratio: float = 0.25,
    max_train_positives: int | None = None,
    skip_missing_labels: bool = False,
    source_revision: str | None = None,
) -> dict:
    """Prepara un dataset auditable. Aborta ante errores críticos de WAID."""
    source = source.resolve()
    output = output.resolve()
    _check_paths(source, output)

    if not math.isfinite(train_negative_ratio) or not 0 <= train_negative_ratio <= 10:
        raise ValueError("--train-negative-ratio debe estar entre 0 y 10")
    if max_train_positives is not None and max_train_positives <= 0:
        raise ValueError("--max-train-positives debe ser mayor que cero")

    audit = audit_dataset(source, hash_check=True)
    if not audit.ok:
        preview = " | ".join(audit.errors[:3])
        raise ValueError(f"WAID no pasó auditoría: {preview} ({len(audit.errors)} errores)")
    total_missing = sum(info.missing_labels for info in audit.splits.values())
    if total_missing and not skip_missing_labels:
        raise ValueError(
            f"Hay {total_missing} imágenes sin TXT. Revisarlas o utilizar "
            "--skip-missing-labels para excluirlas expresamente."
        )

    selected: dict[str, list[ImageRecord]] = {}
    summaries: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        records, skipped = _records(source, split)
        positives = [record for record in records if record.has_cattle]
        negatives = [record for record in records if not record.has_cattle]
        if not positives:
            raise ValueError(f"Split {split} sin imágenes positivas para cattle")

        if split == "train":
            if max_train_positives is not None:
                positives = _stable_sample(positives, max_train_positives, seed, split)
            neg_limit = math.floor(len(positives) * train_negative_ratio)
            negatives = _stable_sample(negatives, neg_limit, seed, split)
        # valid y test retienen todos los ejemplos etiquetados, incluidas otras especies.
        chosen = sorted([*positives, *negatives], key=lambda record: record.image.name)
        selected[split] = chosen
        summaries[split] = {
            "images": len(chosen),
            "positive_images": len(positives),
            "negative_images": len(negatives),
            "cattle_boxes": sum(len(record.cattle_lines) for record in chosen),
            "skipped_missing_labels": skipped,
        }

    manifest: dict = {
        "source": "WAID — https://github.com/xiaohuicui/WAID",
        "source_directory": str(source),
        "source_revision": source_revision,
        "source_class": {"id": CATTLE_ID, "name": "cattle"},
        "target_class": {"id": 0, "name": "cattle"},
        "splits_preserved": list(SPLITS),
        "audit_sha256_across_splits": True,
        "near_duplicate_review": "pending_manual_review",
        "license": "not_verified_for_original_source",
        "parameters": {
            "seed": seed,
            "train_negative_ratio": train_negative_ratio,
            "max_train_positives": max_train_positives,
            "skip_missing_labels": skip_missing_labels,
        },
        "splits": summaries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".waid-cattle-", dir=output.parent))
    try:
        for split, records in selected.items():
            images_dir = temporary / "images" / split
            labels_dir = temporary / "labels" / split
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            for record in records:
                shutil.copy2(record.image, images_dir / record.image.name)
                text = "\n".join(record.cattle_lines)
                (labels_dir / f"{record.image.stem}.txt").write_text(
                    f"{text}\n" if text else "", encoding="utf-8"
                )
            result = validate_labels(labels_dir)
            if result.errors:
                raise ValueError(f"Etiquetas procesadas inválidas: {result.errors[:3]}")

        # Escribir la ruta FINAL, no la carpeta temporal de construcción.
        root_yaml = json.dumps(output.as_posix(), ensure_ascii=False)
        (temporary / "dataset.yaml").write_text(
            f"path: {root_yaml}\n"
            "train: images/train\n"
            "val: images/valid\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n  0: cattle\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WAID → YOLO cattle=0, con auditoría previa")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/waid-cattle"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-negative-ratio", type=float, default=0.25)
    parser.add_argument("--max-train-positives", type=int, default=None)
    parser.add_argument("--skip-missing-labels", action="store_true")
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args(argv)

    try:
        manifest = prepare_dataset(
            args.source,
            args.output,
            seed=args.seed,
            train_negative_ratio=args.train_negative_ratio,
            max_train_positives=args.max_train_positives,
            skip_missing_labels=args.skip_missing_labels,
            source_revision=args.source_revision,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Dataset preparado en: {args.output.resolve()}")
    for split, info in manifest["splits"].items():
        print(f"{split}: {info['images']} imágenes, {info['cattle_boxes']} cajas cattle")
    print(f"Configuración: {args.output / 'dataset.yaml'}")
    print("Pendiente: inspección visual y revisión de similitud entre imágenes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
