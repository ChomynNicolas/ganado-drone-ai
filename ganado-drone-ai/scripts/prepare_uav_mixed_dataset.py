"""Prepara una mezcla local WAID + ganado UAV desde anotaciones Supervisely.

Solo incorpora fotografías con cajas cattle verificadas. Las imágenes sin objetos
se excluyen porque el paquete las marca sin anotaciones, no como negativos.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_IMAGES = 656
EXPECTED_LABELED_IMAGES = 316
EXPECTED_BOXES = 1887


def _cattle_boxes(annotation: dict[str, Any], width: int, height: int) -> list[list[float]]:
    boxes = []
    for index, item in enumerate(annotation.get("objects", []), 1):
        if item.get("classTitle") != "cattle":
            continue
        if item.get("geometryType") != "rectangle":
            raise ValueError(f"El objeto cattle {index} no es una caja rectangular")
        exterior = item.get("points", {}).get("exterior", [])
        if len(exterior) != 2 or any(len(point) != 2 for point in exterior):
            raise ValueError(f"El objeto cattle {index} no tiene dos esquinas válidas")
        x1, y1 = map(float, exterior[0])
        x2, y2 = map(float, exterior[1])
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            raise ValueError(f"El objeto cattle {index} contiene coordenadas no finitas")
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Caja cattle fuera de la imagen {width}x{height}: {(x1, y1, x2, y2)}")
        center_x = (left + right) / (2 * width)
        center_y = (top + bottom) / (2 * height)
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        boxes.append([center_x, center_y, box_width, box_height])
    return boxes


def _animal_ids(annotation: dict[str, Any]) -> set[str]:
    result = set()
    for item in annotation.get("objects", []):
        for tag in item.get("tags", []):
            if tag.get("name") == "cattle id" and tag.get("value") is not None:
                result.add(str(tag["value"]))
    return result


def inspect_source(
    source: Path, *, allow_partial: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir, annotation_dir = source / "img", source / "ann"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Se esperan carpetas img y ann dentro de {source}")
    license_path = source.parent / "LICENSE.md"
    if not license_path.is_file() or "CC BY-NC-ND 4.0" not in license_path.read_text(
        encoding="utf-8"
    ):
        raise ValueError("No se pudo confirmar LICENSE.md con licencia CC BY-NC-ND 4.0")

    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    annotations = {path.name.removesuffix(".json"): path for path in annotation_dir.glob("*.json")}
    if not images:
        raise ValueError(f"No hay imágenes en {image_dir}")
    if {path.name for path in images} != set(annotations):
        missing = sorted({path.name for path in images} - set(annotations))
        extra = sorted(set(annotations) - {path.name for path in images})
        raise ValueError(
            f"Imágenes y anotaciones no coinciden: faltan={missing[:5]}, sobran={extra[:5]}"
        )

    records = []
    id_flights: dict[str, set[str]] = defaultdict(set)
    box_count = 0
    for image_path in images:
        annotation_path = annotations[image_path.name]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        recorded_size = annotation.get("size", {})
        if recorded_size and (
            recorded_size.get("width") != width or recorded_size.get("height") != height
        ):
            raise ValueError(f"Dimensiones de etiqueta e imagen no coinciden: {image_path.name}")
        boxes = _cattle_boxes(annotation, width, height)
        if not boxes:
            continue
        flight = image_path.stem.split("_", 1)[0]
        for animal_id in _animal_ids(annotation):
            id_flights[animal_id].add(flight)
        box_count += len(boxes)
        records.append({"image": image_path, "boxes": boxes, "flight": flight})

    stats = {
        "images_total": len(images),
        "images_with_cattle_boxes": len(records),
        "images_without_cattle_boxes_excluded": len(images) - len(records),
        "cattle_boxes": box_count,
        "distinct_cattle_ids": len(id_flights),
        "cattle_ids_repeated_between_flights": sum(
            len(flights) > 1 for flights in id_flights.values()
        ),
        "license": "CC BY-NC-ND 4.0",
    }
    expected = (EXPECTED_IMAGES, EXPECTED_LABELED_IMAGES, EXPECTED_BOXES)
    actual = (stats["images_total"], stats["images_with_cattle_boxes"], stats["cattle_boxes"])
    if not allow_partial and actual != expected:
        raise ValueError(
            "Dataset 1 incompleto o cambiado: se esperaban 656 imágenes, 316 etiquetadas y "
            f"1887 cajas; se encontraron {actual}. Para una prueba parcial use --allow-partial."
        )
    return records, stats


def _quote_yaml_path(path: Path) -> str:
    return json.dumps(path.resolve().as_posix(), ensure_ascii=False)


def prepare_mixed_dataset(
    source: Path,
    waid: Path,
    output: Path,
    *,
    seed: int = 42,
    waid_share: float = 0.8,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if not 0 < waid_share < 1:
        raise ValueError("waid_share debe estar entre 0 y 1")
    if output.exists():
        raise FileExistsError(f"La carpeta destino ya existe y no se modificará: {output}")
    waid_yaml = waid / "dataset.yaml"
    waid_train_dir = waid / "images" / "train"
    waid_valid_dir = waid / "images" / "valid"
    waid_test_dir = waid / "images" / "test"
    if (
        not waid_yaml.is_file()
        or not waid_train_dir.is_dir()
        or not waid_valid_dir.is_dir()
        or not waid_test_dir.is_dir()
    ):
        raise FileNotFoundError(f"Dataset WAID procesado incompleto: {waid}")
    records, stats = inspect_source(source, allow_partial=allow_partial)
    waid_images = sorted(
        path
        for path in waid_train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not waid_images:
        raise ValueError(f"No hay imágenes WAID en {waid_train_dir}")
    uav_entries = max(1, round(len(waid_images) * (1 - waid_share) / waid_share))
    waid_entries = max(len(waid_images), round(uav_entries * waid_share / (1 - waid_share)))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        uav_image_dir = output / "images" / "uav_train"
        uav_label_dir = output / "labels" / "uav_train"
        uav_image_dir.mkdir(parents=True)
        uav_label_dir.mkdir(parents=True)
        staged_records = []
        for record in records:
            source_image = record["image"]
            destination_name = source_image.name
            if (uav_image_dir / destination_name).exists():
                destination_name = f"{record['flight']}__{destination_name}"
            shutil.copy2(source_image, uav_image_dir / destination_name)
            label_path = uav_label_dir / f"{Path(destination_name).stem}.txt"
            label_path.write_text(
                "".join(
                    "0 " + " ".join(f"{value:.8f}" for value in box) + "\n"
                    for box in record["boxes"]
                ),
                encoding="utf-8",
            )
            staged_records.append({**record, "staged_image": uav_image_dir / destination_name})

        rng = random.Random(seed)
        candidate_paths = [record["staged_image"] for record in staged_records]
        shuffled = candidate_paths.copy()
        rng.shuffle(shuffled)
        repeated_paths = [shuffled[index % len(shuffled)] for index in range(uav_entries)]
        repeated_waid = [waid_images[index % len(waid_images)] for index in range(waid_entries)]
        train_list = [path.resolve() for path in repeated_waid] + [
            path.resolve() for path in repeated_paths
        ]
        rng.shuffle(train_list)
        (output / "train.txt").write_text(
            "".join(f"{path.as_posix()}\n" for path in train_list), encoding="utf-8"
        )
        waid_valid = waid_valid_dir.resolve()
        waid_test = waid_test_dir.resolve()
        yaml_text = (
            f"path: {_quote_yaml_path(output)}\n"
            f"train: {_quote_yaml_path(output / 'train.txt')}\n"
            f"val: {_quote_yaml_path(waid_valid)}\n"
            f"test: {_quote_yaml_path(waid_test)}\n"
            "nc: 1\nnames:\n  0: cattle\n"
        )
        (output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
        manifest = {
            "dataset": "WAID 5000 + Cattle Detection and Counting in UAV Images Dataset1",
            "protocol": "80% WAID, 20% UAV by image exposures; UAV Dataset1 train only",
            "seed": seed,
            "waid_share_target": waid_share,
            "waid_train_images": len(waid_images),
            "waid_train_image_exposures": waid_entries,
            "uav_train_images_unique": len(staged_records),
            "uav_train_image_exposures": uav_entries,
            "train_image_exposures_total": len(train_list),
            "waid_share_actual": waid_entries / len(train_list),
            "uav_share_actual": uav_entries / len(train_list),
            "validation": "WAID valid only; Dataset1 is not split because cattle IDs recur across flights",
            "test": "WAID test, Dataset2 and NZ cow_images remain reserved for final evaluation",
            "source_statistics": stats,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mezcla WAID con cajas UAV verificadas")
    parser.add_argument(
        "--source", type=Path, required=True, help="Carpeta Dataset1 con img/ y ann/"
    )
    parser.add_argument("--waid", type=Path, default=Path("data/processed/waid-cattle-5000-v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/waid-cattle-uav-mixed-v1")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--waid-share", type=float, default=0.8)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_mixed_dataset(
            args.source.resolve(),
            args.waid.resolve(),
            args.output.resolve(),
            seed=args.seed,
            waid_share=args.waid_share,
            allow_partial=args.allow_partial,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"Dataset preparado: WAID={manifest['waid_train_images']} + "
        f"UAV único={manifest['uav_train_images_unique']} / exposiciones UAV="
        f"{manifest['uav_train_image_exposures']} ({manifest['uav_share_actual']:.1%})."
    )
    print(f"Manifest: {(args.output / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
