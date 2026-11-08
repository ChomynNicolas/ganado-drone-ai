from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_uav_mixed_dataset import _cattle_boxes, inspect_source, prepare_mixed_dataset


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "Dataset1"
    (source / "img").mkdir(parents=True)
    (source / "ann").mkdir()
    (source.parent / "LICENSE.md").write_text("CC BY-NC-ND 4.0", encoding="utf-8")
    Image.new("RGB", (100, 80)).save(source / "img" / "auto1_DJI_0001.JPG")
    annotation = {
        "size": {"width": 100, "height": 80},
        "objects": [
            {
                "classTitle": "cattle",
                "geometryType": "rectangle",
                "points": {"exterior": [[10, 20], [30, 40]]},
                "tags": [{"name": "cattle id", "value": 1}],
            }
        ],
    }
    (source / "ann" / "auto1_DJI_0001.JPG.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    Image.new("RGB", (100, 80)).save(source / "img" / "auto1_DJI_0002.JPG")
    (source / "ann" / "auto1_DJI_0002.JPG.json").write_text(
        json.dumps({"size": {"width": 100, "height": 80}, "objects": []}),
        encoding="utf-8",
    )
    return source


def _waid(tmp_path: Path) -> Path:
    root = tmp_path / "waid"
    for split in ("train", "valid", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "train" / "waid.jpg")
    (root / "labels" / "train" / "waid.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/valid\ntest: images/test\n"
    )
    return root


def test_cattle_boxes_converts_supervisely_rectangles_to_yolo() -> None:
    boxes = _cattle_boxes(
        {
            "objects": [
                {
                    "classTitle": "cattle",
                    "geometryType": "rectangle",
                    "points": {"exterior": [[10, 20], [30, 40]]},
                }
            ]
        },
        100,
        80,
    )
    assert boxes == [[0.2, 0.375, 0.2, 0.25]]


def test_inspection_excludes_unlabeled_images_and_counts_verified_boxes(tmp_path: Path) -> None:
    records, stats = inspect_source(_source(tmp_path), allow_partial=True)
    assert len(records) == 1
    assert stats["images_without_cattle_boxes_excluded"] == 1
    assert stats["cattle_boxes"] == 1


def test_prepare_mixed_dataset_builds_80_20_exposure_manifest(tmp_path: Path) -> None:
    output = tmp_path / "processed" / "mixed"
    manifest = prepare_mixed_dataset(_source(tmp_path), _waid(tmp_path), output, allow_partial=True)
    assert manifest["uav_train_image_exposures"] == 1
    assert manifest["uav_share_actual"] == pytest.approx(0.2)
    assert manifest["uav_train_images_unique"] == 1
    assert (output / "labels" / "uav_train" / "auto1_DJI_0001.txt").is_file()
    assert (output / "dataset.yaml").is_file()
    with pytest.raises(FileExistsError):
        prepare_mixed_dataset(_source(tmp_path), _waid(tmp_path), output, allow_partial=True)


def test_inspection_rejects_unconfirmed_license(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source.parent / "LICENSE.md").write_text("unknown", encoding="utf-8")
    with pytest.raises(ValueError, match="CC BY-NC-ND"):
        inspect_source(source, allow_partial=True)
