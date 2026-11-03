"""Pruebas de la inspección visual generada con Pillow."""

from pathlib import Path

import pytest
from PIL import Image

from scripts.preview_cattle import create_preview, main


def _processed_fixture(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    images = dataset / "images" / "train"
    labels = dataset / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for n in range(3):
        Image.new("RGB", (80, 60), (100, 160, n * 70)).save(images / f"{n}.jpg")
        (labels / f"{n}.txt").write_text("0 0.5 0.5 0.4 0.3\n" if n != 2 else "", encoding="utf-8")
    return dataset


def test_contact_sheet_is_generated(tmp_path: Path) -> None:
    dataset = _processed_fixture(tmp_path)
    target = tmp_path / "preview.jpg"
    assert create_preview(dataset, target, count=3, columns=2) == 3
    with Image.open(target) as sheet:
        assert sheet.size == (720, 780)


def test_invalid_split_or_count(tmp_path: Path) -> None:
    dataset = _processed_fixture(tmp_path)
    with pytest.raises(ValueError, match="split inválido"):
        create_preview(dataset, tmp_path / "preview.jpg", split="oops")
    with pytest.raises(ValueError, match="positivos"):
        create_preview(dataset, tmp_path / "preview.jpg", count=0)


def test_invalid_label_is_rejected(tmp_path: Path) -> None:
    dataset = _processed_fixture(tmp_path)
    (dataset / "labels/train/0.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class_id"):
        create_preview(dataset, tmp_path / "preview.jpg", count=3)


def test_cli_returns_success(tmp_path: Path) -> None:
    dataset = _processed_fixture(tmp_path)
    output = tmp_path / "preview.jpg"
    assert main(["--dataset", str(dataset), "--output", str(output), "--count", "2"]) == 0
    assert output.is_file()
