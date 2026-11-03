"""Pruebas sintéticas de conversión WAID; no sustituyen una validación real."""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_waid import WAID_CLASSES
from scripts.prepare_cattle_dataset import main, prepare_dataset
from scripts.validate_yolo_labels import validate_labels


def _fake_waid(
    tmp_path: Path, *, positives: int = 3, negatives: int = 2, missing: bool = False
) -> Path:
    source = tmp_path / "WAID"
    source.mkdir()
    (source / "classes.txt").write_text("\n".join(WAID_CLASSES), encoding="utf-8")
    for split_index, split in enumerate(("train", "valid", "test")):
        img_dir = source / "images" / split
        label_dir = source / "labels" / split
        img_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for i in range(positives + negatives):
            stem = f"{split}-{i:02d}"
            Image.new("RGB", (48, 32), (split_index * 70, i * 14, 120)).save(
                img_dir / f"{stem}.jpg"
            )
            if missing and split == "test" and i == 0:
                continue
            label = (
                "1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n" if i < positives else "0 0.5 0.5 0.3 0.3\n"
            )
            (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
    return source


def test_convert_class_and_keep_original_splits(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    original = (raw / "labels/train/train-00.txt").read_text(encoding="utf-8")
    out = tmp_path / "output"
    manifest = prepare_dataset(raw, out, train_negative_ratio=1)

    assert (raw / "labels/train/train-00.txt").read_text(encoding="utf-8") == original
    assert (out / "labels/train/train-00.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.2\n"
    )
    assert manifest["splits"]["train"] == {
        "images": 5,
        "positive_images": 3,
        "negative_images": 2,
        "cattle_boxes": 3,
        "skipped_missing_labels": 0,
    }
    assert manifest["splits"]["valid"]["images"] == 5
    assert manifest["splits"]["test"]["images"] == 5
    assert validate_labels(out / "labels", num_classes=1).errors == ()
    assert (out / "labels/test/test-04.txt").read_text(encoding="utf-8") == ""
    assert "names:\n  0: cattle\n" in (out / "dataset.yaml").read_text(encoding="utf-8")
    assert f'path: "{out.as_posix()}"' in (out / "dataset.yaml").read_text(encoding="utf-8")
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_sample_only_train_retains_val_and_test(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path, positives=5, negatives=5)
    manifest = prepare_dataset(
        raw, tmp_path / "output", max_train_positives=2, train_negative_ratio=0.5
    )
    assert manifest["splits"]["train"]["images"] == 3
    assert manifest["splits"]["train"]["positive_images"] == 2
    assert manifest["splits"]["train"]["negative_images"] == 1
    assert manifest["splits"]["valid"]["images"] == 10
    assert manifest["splits"]["test"]["images"] == 10


def test_sample_is_deterministic(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path, positives=9, negatives=9)
    settings = {"seed": 21, "max_train_positives": 4, "train_negative_ratio": 0.5}
    prepare_dataset(raw, tmp_path / "out-1", **settings)
    prepare_dataset(raw, tmp_path / "out-2", **settings)
    one = {p.name for p in (tmp_path / "out-1/images/train").iterdir()}
    two = {p.name for p in (tmp_path / "out-2/images/train").iterdir()}
    assert one == two


def test_source_duplicates_across_splits_fail(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    duplicate = (raw / "images/train/train-00.jpg").read_bytes()
    (raw / "images/test/test-00.jpg").write_bytes(duplicate)
    with pytest.raises(ValueError, match="no pasó auditoría"):
        prepare_dataset(raw, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_invalid_boxes_fail_preflight(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    (raw / "labels/train/train-00.txt").write_text("1 0.9 0.9 0.8 0.8", encoding="utf-8")
    with pytest.raises(ValueError, match="no pasó auditoría"):
        prepare_dataset(raw, tmp_path / "output")


def test_missing_labels_are_rejected_by_default(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path, missing=True)
    with pytest.raises(ValueError, match="sin TXT"):
        prepare_dataset(raw, tmp_path / "output")


def test_missing_labels_can_be_explicitly_skipped(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path, missing=True)
    manifest = prepare_dataset(raw, tmp_path / "output", skip_missing_labels=True)
    assert manifest["splits"]["test"]["skipped_missing_labels"] == 1
    assert manifest["splits"]["test"]["images"] == 4


def test_no_overwriting_or_nested_output(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="ya existe"):
        prepare_dataset(raw, existing)
    with pytest.raises(ValueError, match="contenerse"):
        prepare_dataset(raw, raw / "processed")


@pytest.mark.parametrize("ratio", [-0.5, 11, float("nan"), float("inf")])
def test_invalid_negative_ratio(tmp_path: Path, ratio: float) -> None:
    raw = _fake_waid(tmp_path)
    with pytest.raises(ValueError, match="train-negative-ratio"):
        prepare_dataset(raw, tmp_path / "output", train_negative_ratio=ratio)


def test_invalid_max_train_positives(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    with pytest.raises(ValueError, match="max-train-positives"):
        prepare_dataset(raw, tmp_path / "output", max_train_positives=0)


def test_no_positives_in_split_fails(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    for label in (raw / "labels" / "valid").glob("*.txt"):
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sin imágenes positivas"):
        prepare_dataset(raw, tmp_path / "output")


def test_cli_writes_dataset(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    out = tmp_path / "output"
    assert main(["--source", str(raw), "--output", str(out), "--source-revision", "abc"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_revision"] == "abc"
    assert main(["--source", str(raw), "--output", str(out)]) == 1
