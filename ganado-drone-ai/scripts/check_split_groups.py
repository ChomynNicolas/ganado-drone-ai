"""Verifica que una fotografía original no aparezca en múltiples splits.

Agrupa imágenes según el nombre previo a `.rf.`. No detecta escenas
casi idénticas de identificadores distintos (p.ej., fotogramas vecinos).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.audit_waid import IMAGE_SUFFIXES, SPLITS


def check_split_groups(dataset: Path) -> dict:
    """Lee nombres de archivos, sin modificar imágenes o anotaciones."""
    appearances: dict[str, set[str]] = defaultdict(set)
    images_per_split: dict[str, int] = {}
    groups_per_split: dict[str, int] = {}

    for split in SPLITS:
        image_dir = dataset / "images" / split
        if not image_dir.is_dir():
            raise ValueError(f"No existe el directorio de imágenes: {image_dir}")
        names: set[str] = set()
        count = 0
        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            count += 1
            source_id = image.stem.split(".rf.", 1)[0]
            names.add(source_id)
            appearances[source_id].add(split)
        images_per_split[split] = count
        groups_per_split[split] = len(names)

    if sum(images_per_split.values()) == 0:
        raise ValueError(f"No se encontraron imágenes en: {dataset}")
    overlaps = {
        source_id: sorted(splits)
        for source_id, splits in sorted(appearances.items())
        if len(splits) > 1
    }
    return {
        "dataset": str(dataset.resolve()),
        "images_per_split": images_per_split,
        "groups_per_split": groups_per_split,
        "overlapping_source_groups": overlaps,
        "overlap_count": len(overlaps),
        "ok": not overlaps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Comprueba grupos originales entre splits")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_split_groups(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    for split in SPLITS:
        print(
            f"{split}: {result['images_per_split'][split]} imágenes / "
            f"{result['groups_per_split'][split]} grupos de origen"
        )
    print(f"Grupos en varios splits: {result['overlap_count']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Reporte: {args.output}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
