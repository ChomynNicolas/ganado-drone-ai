"""Convierte WAID (cattle=1) a un dataset YOLO monocategoría (cattle=0).

Conserva train/valid/test sin mezclarlos y se niega a escribir sobre datos
procesados existentes. No modifica los archivos de origen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
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
    def source_id(self) -> str:
        """ID anterior a la marca de exportación .rf.<hash>."""
        return self.image.stem.split(".rf.", 1)[0]

    @property
    def has_cattle(self) -> bool:
        return bool(self.cattle_lines)


def _is_sheep_zero_box(raw: str) -> bool:
    """Admite únicamente cajas sheep con ancho/alto exactamente cero.

    Las demás coordenadas deben ser numéricas, finitas y estar normalizadas.
    """
    fields = raw.split()
    if len(fields) != 5 or fields[0] != "0":
        return False
    try:
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError:
        return False
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return False
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return False
    if not (0 <= width <= 1 and 0 <= height <= 1):
        return False
    if width != 0 and height != 0:
        return False
    return (
        x - width / 2 >= -1e-6
        and x + width / 2 <= 1 + 1e-6
        and y - height / 2 >= -1e-6
        and y + height / 2 <= 1 + 1e-6
    )


def _allowed_source_error(error: str) -> bool:
    """Clasifica estrictamente el error del auditor que podemos descartar."""
    match = re.fullmatch(
        r"(?P<path>.+\.txt):(?P<line>\d+): "
        r"width y height deben estar en \(0, 1]",
        error,
    )
    if match is None:
        return False
    path = Path(match.group("path"))
    try:
        raw = path.read_text(encoding="utf-8").splitlines()[int(match.group("line")) - 1]
    except (OSError, UnicodeError, IndexError):
        return False
    return _is_sheep_zero_box(raw)


def _read_cattle_lines(
    label_path: Path, *, allow_invalid_sheep_zero_boxes: bool = False
) -> tuple[str, ...]:
    lines: list[str] = []
    for number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        if allow_invalid_sheep_zero_boxes and _is_sheep_zero_box(raw):
            continue
        try:
            class_id, *_ = validate_yolo_line(raw, num_classes=len(WAID_CLASSES))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{number}: {exc}") from exc
        if class_id == CATTLE_ID:
            # Mantener las coordenadas originales sin redondear nuevamente.
            coordinates = raw.split()[1:]
            lines.append("0 " + " ".join(coordinates))
    return tuple(lines)


def _source_group_overlaps(records_by_split: dict[str, list[ImageRecord]]) -> dict[str, list[str]]:
    appearances: dict[str, set[str]] = defaultdict(set)
    for split, records in records_by_split.items():
        for record in records:
            appearances[record.source_id].add(split)
    return {
        source_id: sorted(splits) for source_id, splits in appearances.items() if len(splits) > 1
    }


def _group_resplit(
    records_by_split: dict[str, list[ImageRecord]], seed: int
) -> tuple[dict[str, list[ImageRecord]], dict[str, int]]:
    """Resplit determinista estratificado por presencia de cattle, agrupado por foto.

    La estratificación se hace por GRUPO, no por imagen; los tamaños de los
    splits pueden apartarse de 70/20/10 según el tamaño de los grupos.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    filenames: set[str] = set()
    for split in SPLITS:
        for record in records_by_split[split]:
            if record.image.name in filenames:
                raise ValueError(f"Nombre de imagen repetido entre splits: {record.image.name}")
            filenames.add(record.image.name)
            groups[record.source_id].append(record)

    cohorts: dict[bool, list[str]] = {True: [], False: []}
    for source_id, records in groups.items():
        cohorts[any(record.has_cattle for record in records)].append(source_id)

    assignments: dict[str, str] = {}
    group_counts: dict[str, int] = {split: 0 for split in SPLITS}
    for has_cattle, source_ids in cohorts.items():
        if len(source_ids) < 3:
            raise ValueError(
                f"No hay grupos suficientes para repartir positivos={has_cattle}: "
                f"{len(source_ids)}; se requieren al menos 3"
            )
        ranked = sorted(
            source_ids,
            key=lambda name: (
                hashlib.sha256(f"{seed}:{has_cattle}:{name}".encode()).digest(),
                name,
            ),
        )
        count = len(ranked)
        n_valid = max(1, round(count * 0.20))
        n_test = max(1, round(count * 0.10))
        n_train = count - n_valid - n_test
        if n_train < 1:
            n_train, n_valid, n_test = 1, 1, count - 2
        for split, ids in zip(
            SPLITS,
            (ranked[:n_train], ranked[n_train : n_train + n_valid], ranked[-n_test:]),
        ):
            for source_id in ids:
                assignments[source_id] = split
                group_counts[split] += 1

    selected: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}
    for source_id, records in groups.items():
        selected[assignments[source_id]].extend(records)
    return selected, group_counts


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


def _records(
    source: Path, split: str, *, allow_invalid_sheep_zero_boxes: bool = False
) -> tuple[list[ImageRecord], int]:
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
        records.append(
            ImageRecord(
                image=image,
                cattle_lines=_read_cattle_lines(
                    label, allow_invalid_sheep_zero_boxes=allow_invalid_sheep_zero_boxes
                ),
            )
        )
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
    allow_invalid_sheep_zero_boxes: bool = False,
    resplit_by_source_image: bool = False,
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
    accepted_errors = (
        [issue for issue in audit.errors if _allowed_source_error(issue)]
        if allow_invalid_sheep_zero_boxes
        else []
    )
    rejected_errors = [issue for issue in audit.errors if issue not in accepted_errors]
    if rejected_errors:
        preview = " | ".join(rejected_errors[:3])
        raise ValueError(
            f"WAID no pasó auditoría: {preview} ({len(rejected_errors)} errores no admitidos)"
        )
    total_missing = sum(info.missing_labels for info in audit.splits.values())
    if total_missing and not skip_missing_labels:
        raise ValueError(
            f"Hay {total_missing} imágenes sin TXT. Revisarlas o utilizar "
            "--skip-missing-labels para excluirlas expresamente."
        )

    records_by_source_split: dict[str, list[ImageRecord]] = {}
    skipped_by_split: dict[str, int] = {}
    for split in SPLITS:
        records, skipped = _records(
            source, split, allow_invalid_sheep_zero_boxes=allow_invalid_sheep_zero_boxes
        )
        records_by_source_split[split] = records
        skipped_by_split[split] = skipped

    overlaps = _source_group_overlaps(records_by_source_split)
    if overlaps and not resplit_by_source_image:
        raise ValueError(
            f"Hay {len(overlaps)} identificadores de fotografía original en varios "
            "splits; utilizar --resplit-by-source-image para evitar ese leakage"
        )
    if resplit_by_source_image:
        assigned, group_counts = _group_resplit(records_by_source_split, seed)
    else:
        assigned = records_by_source_split
        group_counts = {
            split: len({record.source_id for record in records})
            for split, records in assigned.items()
        }

    selected: dict[str, list[ImageRecord]] = {}
    summaries: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        records = assigned[split]
        positives = [record for record in records if record.has_cattle]
        negatives = [record for record in records if not record.has_cattle]
        if not positives:
            raise ValueError(f"Split {split} sin imágenes positivas para cattle")

        if split == "train":
            if max_train_positives is not None:
                positives = _stable_sample(positives, max_train_positives, seed, split)
            neg_limit = math.floor(len(positives) * train_negative_ratio)
            negatives = _stable_sample(negatives, neg_limit, seed, split)
        chosen = sorted([*positives, *negatives], key=lambda record: record.image.name)
        selected[split] = chosen
        summaries[split] = {
            "images": len(chosen),
            "positive_images": len(positives),
            "negative_images": len(negatives),
            "cattle_boxes": sum(len(record.cattle_lines) for record in chosen),
            "skipped_missing_labels": skipped_by_split[split],
            "source_groups_assigned": group_counts[split],
        }

    manifest: dict = {
        "source": "WAID — https://github.com/xiaohuicui/WAID",
        "source_directory": str(source),
        "source_revision": source_revision,
        "source_class": {"id": CATTLE_ID, "name": "cattle"},
        "target_class": {"id": 0, "name": "cattle"},
        "split_policy": (
            "grouped_source_image_70_20_10" if resplit_by_source_image else "original_verified"
        ),
        "original_source_ids_crossing_splits": len(overlaps),
        "excluded_invalid_sheep_zero_boxes": accepted_errors,
        "excluded_invalid_sheep_zero_boxes_count": len(accepted_errors),
        "audit_sha256_across_splits": True,
        "near_duplicate_review": "pending_manual_review_by_flight_or_scene",
        "license": "not_verified_for_original_source",
        "parameters": {
            "seed": seed,
            "train_negative_ratio": train_negative_ratio,
            "max_train_positives": max_train_positives,
            "skip_missing_labels": skip_missing_labels,
            "allow_invalid_sheep_zero_boxes": allow_invalid_sheep_zero_boxes,
            "resplit_by_source_image": resplit_by_source_image,
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
    parser.add_argument("--allow-invalid-sheep-zero-boxes", action="store_true")
    parser.add_argument("--resplit-by-source-image", action="store_true")
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
            allow_invalid_sheep_zero_boxes=args.allow_invalid_sheep_zero_boxes,
            resplit_by_source_image=args.resplit_by_source_image,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Dataset preparado en: {args.output.resolve()}")
    for split, info in manifest["splits"].items():
        print(f"{split}: {info['images']} imágenes, {info['cattle_boxes']} cajas cattle")
    print(f"Configuración: {args.output / 'dataset.yaml'}")
    print(
        f"Cajas sheep degeneradas excluidas: {manifest['excluded_invalid_sheep_zero_boxes_count']}"
    )
    print("Pendiente: inspección visual y revisión de similitud por vuelo/escena.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
