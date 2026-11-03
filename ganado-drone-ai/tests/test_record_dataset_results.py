"""Pruebas de trazabilidad sobre manifiestos mínimos artificiales."""

import json
from pathlib import Path

import pytest

from scripts.record_dataset_results import build_report, main


def evidence():
    manifest = {
        "source_revision": "a" * 40,
        "split_policy": "grouped_source_image_70_20_10",
        "excluded_invalid_sheep_zero_boxes_count": 51,
        "original_source_ids_crossing_splits": 788,
        "parameters": {
            "seed": 42,
            "train_negative_ratio": 0.25,
            "max_train_positives": 500,
            "allow_invalid_sheep_zero_boxes": True,
        },
        "splits": {
            "train": {
                "images": 3,
                "positive_images": 2,
                "negative_images": 1,
                "cattle_boxes": 4,
            },
            "valid": {
                "images": 2,
                "positive_images": 1,
                "negative_images": 1,
                "cattle_boxes": 2,
            },
            "test": {
                "images": 2,
                "positive_images": 2,
                "negative_images": 0,
                "cattle_boxes": 3,
            },
        },
    }
    groups = {
        "ok": True,
        "overlap_count": 0,
        "images_per_split": {"train": 3, "valid": 2, "test": 2},
        "groups_per_split": {"train": 2, "valid": 1, "test": 2},
    }
    return manifest, groups


def test_build_report_includes_revision_and_totals_without_local_paths():
    manifest, groups = evidence()
    manifest["source_directory"] = r"C:\Users\private\dataset"
    text = build_report(manifest, groups)
    assert "**Total** | **7**" in text
    assert "**9**" in text
    assert "a" * 40 in text
    assert "C:\\Users" not in text
    assert "pendiente" in text


def test_rejects_missing_revision():
    manifest, groups = evidence()
    manifest["source_revision"] = None
    with pytest.raises(ValueError, match="revisión Git"):
        build_report(manifest, groups)


def test_rejects_leakage():
    manifest, groups = evidence()
    groups["overlap_count"] = 1
    groups["ok"] = False
    with pytest.raises(ValueError, match="no pasó"):
        build_report(manifest, groups)


def test_rejects_mismatched_split_counts():
    manifest, groups = evidence()
    groups["images_per_split"]["train"] = 4
    with pytest.raises(ValueError, match="no coincide"):
        build_report(manifest, groups)


def test_rejects_inconsistent_positive_negative_counts():
    manifest, groups = evidence()
    manifest["splits"]["train"]["negative_images"] = 2
    with pytest.raises(ValueError, match="inconsistentes"):
        build_report(manifest, groups)


def test_cli_writes_markdown(tmp_path: Path):
    manifest, groups = evidence()
    m = tmp_path / "manifest.json"
    g = tmp_path / "groups.json"
    output = tmp_path / "output.md"
    m.write_text(json.dumps(manifest), encoding="utf-8")
    g.write_text(json.dumps(groups), encoding="utf-8")
    assert main(["--manifest", str(m), "--groups", str(g), "--output", str(output)]) == 0
    assert "grupo" in output.read_text(encoding="utf-8")
