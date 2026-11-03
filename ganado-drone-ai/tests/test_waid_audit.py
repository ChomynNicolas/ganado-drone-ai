"""Pruebas sintéticas del auditor; no constituyen auditoría de WAID real."""

from pathlib import Path

from scripts.audit_waid import WAID_CLASSES, audit_dataset, main


def fixture_waid(tmp_path: Path, label: str = "1 0.5 0.5 0.2 0.2\n") -> Path:
    root = tmp_path / "WAID"
    root.mkdir()
    (root / "classes.txt").write_text("\n".join(WAID_CLASSES) + "\n", encoding="utf-8")
    for split in ("train", "valid", "test"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{split}.jpg").write_bytes(f"fake-image-{split}".encode())
        (labels / f"{split}.txt").write_text(label, encoding="utf-8")
    return root


def test_counts_cattle_original_id_one(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path)
    result = audit_dataset(root)
    assert result.ok
    assert result.source_cattle_id == 1
    assert result.splits["train"].cattle_boxes == 1
    assert result.splits["test"].cattle_images == 1
    assert result.splits["valid"].classes["sheep"] == 0


def test_mismatched_class_order_is_error(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path)
    (root / "classes.txt").write_text("cattle\nsheep\n", encoding="utf-8")
    result = audit_dataset(root)
    assert not result.ok
    assert any("Mapeo de clases" in item for item in result.errors)


def test_invalid_coordinate_is_error(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path, label="1 0.9 0.9 0.8 0.8\n")
    result = audit_dataset(root)
    assert not result.ok
    assert any("fuera" in item for item in result.errors)


def test_unknown_class_id_is_error(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path, label="7 0.5 0.5 0.2 0.2\n")
    result = audit_dataset(root)
    assert not result.ok
    assert any("class_id 7" in item for item in result.errors)


def test_label_without_image_is_error(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path)
    (root / "labels" / "train" / "orphan.txt").write_text("", encoding="utf-8")
    result = audit_dataset(root)
    assert not result.ok
    assert result.splits["train"].orphan_labels == 1


def test_missing_label_warning_not_error(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path)
    (root / "labels" / "test" / "test.txt").unlink()
    result = audit_dataset(root)
    assert result.ok
    assert result.splits["test"].missing_labels == 1


def test_exact_duplicates_across_splits_only_with_flag(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path)
    (root / "images" / "valid" / "valid.jpg").write_bytes(b"fake-image-train")
    assert audit_dataset(root).ok
    result = audit_dataset(root, hash_check=True)
    assert not result.ok
    assert any("idéntica" in item for item in result.errors)


def test_duplicate_box_is_error(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path, label="1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    result = audit_dataset(root)
    assert not result.ok
    assert result.splits["train"].cattle_boxes == 1


def test_cli_writes_report_and_exit_success(tmp_path: Path) -> None:
    root = fixture_waid(tmp_path)
    output = tmp_path / "report.json"
    code = main(["--dataset", str(root), "--output", str(output), "--hash-check"])
    assert code == 0
    report = output.read_text(encoding="utf-8")
    assert '"source_cattle_id": 1' in report
    assert '"hash_check": true' in report


def test_cli_bad_dataset_fails(tmp_path: Path) -> None:
    code = main(["--dataset", str(tmp_path / "missing")])
    assert code == 1
