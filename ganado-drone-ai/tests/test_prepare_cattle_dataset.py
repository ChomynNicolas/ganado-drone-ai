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
        "source_groups_assigned": 5,
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


def test_sheep_zero_box_requires_explicit_opt_in(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    label = raw / "labels/train/train-00.txt"
    label.write_text("1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no pasó auditoría"):
        prepare_dataset(raw, tmp_path / "should-fail")
    manifest = prepare_dataset(raw, tmp_path / "success", allow_invalid_sheep_zero_boxes=True)
    assert manifest["excluded_invalid_sheep_zero_boxes_count"] == 1
    assert len(manifest["excluded_invalid_sheep_zero_boxes"]) == 1
    assert (tmp_path / "success/labels/train/train-00.txt").read_text() == ("0 0.5 0.5 0.2 0.2\n")
    assert "0 0.2 0.2 0 0.1" in label.read_text()


def test_cattle_zero_box_remains_fatal(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path)
    (raw / "labels/train/train-00.txt").write_text("1 0.5 0.5 0 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no pasó auditoría"):
        prepare_dataset(
            raw,
            tmp_path / "output",
            allow_invalid_sheep_zero_boxes=True,
        )


def test_source_image_overlap_is_blocked_or_reassigned(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path, positives=8, negatives=8)
    train_image = raw / "images/train/train-00.jpg"
    train_label = raw / "labels/train/train-00.txt"
    valid_image = raw / "images/valid/valid-00.jpg"
    valid_label = raw / "labels/valid/valid-00.txt"
    # Misma fotografía base, pero ambas variantes no tienen el mismo hash.
    train_image.rename(train_image.with_name("sceneA.rf.aaa.jpg"))
    train_label.rename(train_label.with_name("sceneA.rf.aaa.txt"))
    valid_image.rename(valid_image.with_name("sceneA.rf.bbb.jpg"))
    valid_label.rename(valid_label.with_name("sceneA.rf.bbb.txt"))

    with pytest.raises(ValueError, match="resplit-by-source-image"):
        prepare_dataset(raw, tmp_path / "blocked")

    out = tmp_path / "grouped"
    manifest = prepare_dataset(raw, out, resplit_by_source_image=True)
    assert manifest["original_source_ids_crossing_splits"] == 1
    assert manifest["split_policy"] == "grouped_source_image_70_20_10"
    assert (out / "dataset.yaml").is_file()

    assignments: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        for image in (out / "images" / split).glob("*.jpg"):
            group = image.stem.split(".rf.")[0]
            assignments.setdefault(group, set()).add(split)
    assert all(len(assigned) == 1 for assigned in assignments.values())
    # Las dos variantes deben quedar juntas si se seleccionan.
    found = [
        split
        for split in ("train", "valid", "test")
        if (out / "images" / split / "sceneA.rf.aaa.jpg").exists()
        and (out / "images" / split / "sceneA.rf.bbb.jpg").exists()
    ]
    assert found or not any(
        (out / "images" / split / "sceneA.rf.aaa.jpg").exists()
        for split in ("train", "valid", "test")
    )


def test_grouped_split_is_reproducible(tmp_path: Path) -> None:
    raw = _fake_waid(tmp_path, positives=10, negatives=10)
    a = tmp_path / "a"
    b = tmp_path / "b"
    prepare_dataset(raw, a, seed=99, resplit_by_source_image=True)
    prepare_dataset(raw, b, seed=99, resplit_by_source_image=True)
    for split in ("train", "valid", "test"):
        images_a = {p.name for p in (a / "images" / split).iterdir()}
        images_b = {p.name for p in (b / "images" / split).iterdir()}
        assert images_a == images_b
