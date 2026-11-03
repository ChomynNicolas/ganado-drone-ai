"""Pruebas unitarias de agrupación sin depender del WAID descargado."""

from pathlib import Path

import pytest

from scripts.check_split_groups import check_split_groups, main


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    for split in ("train", "valid", "test"):
        (dataset / "images" / split).mkdir(parents=True)
    return dataset


def test_clean_groups(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    for split in ("train", "valid", "test"):
        (dataset / "images" / split / f"{split}.rf.abcd.jpg").touch()
    result = check_split_groups(dataset)
    assert result["ok"] is True
    assert result["overlap_count"] == 0
    assert all(number == 1 for number in result["groups_per_split"].values())
    assert main(["--dataset", str(dataset)]) == 0


def test_cross_split_variants_fail(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (dataset / "images/train/source.rf.a.jpg").touch()
    (dataset / "images/valid/source.rf.b.jpg").touch()
    (dataset / "images/test/different.jpg").touch()
    result = check_split_groups(dataset)
    assert result["ok"] is False
    assert result["overlapping_source_groups"] == {"source": ["train", "valid"]}
    assert main(["--dataset", str(dataset)]) == 1


def test_missing_images_folder_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No existe"):
        check_split_groups(tmp_path)


def test_empty_dataset_fails(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError, match="No se encontraron"):
        check_split_groups(dataset)
