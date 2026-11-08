from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_yolo import (
    _match_boxes,
    label_count,
    labels_dir_for_images,
    mean_absolute_error,
    read_dataset_root,
    read_dataset_split,
    select_threshold,
)
from scripts.train_yolo import select_device


def test_select_device_uses_available_cuda_or_cpu() -> None:
    assert select_device("auto", True) == 0
    assert select_device("auto", False) == "cpu"
    assert select_device("cpu", True) == "cpu"


def test_select_device_rejects_missing_requested_cuda() -> None:
    with pytest.raises(ValueError, match="no detecta una GPU"):
        select_device("cuda", False)


def test_mean_absolute_error_computes_per_image_count_error() -> None:
    assert mean_absolute_error([2, 0, 4], [1, 2, 7]) == pytest.approx(2.0)


def test_mean_absolute_error_rejects_empty_or_mismatched_lists() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([], [])
    with pytest.raises(ValueError):
        mean_absolute_error([1], [0, 1])


def test_label_count_counts_only_processed_cattle_class(tmp_path: Path) -> None:
    label = tmp_path / "image.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.3 0.1 0.1\n0 0.1 0.1 0.2 0.2\n")

    assert label_count(label) == 2


def test_read_dataset_root_reads_generated_yaml_path(tmp_path: Path) -> None:
    root = tmp_path / "waid-cattle"
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(f"path: {json.dumps(root.as_posix())}\ntrain: images/train\n")

    assert read_dataset_root(dataset_yaml) == root


def test_read_dataset_split_resolves_absolute_mixed_validation(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    valid = tmp_path / "waid" / "images" / "valid"
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True)
    dataset_yaml.write_text(
        f"path: {json.dumps(root.as_posix())}\nval: {json.dumps(valid.as_posix())}\n"
    )

    assert read_dataset_split(dataset_yaml, root, "val") == valid
    assert labels_dir_for_images(valid) == tmp_path / "waid" / "labels" / "valid"


def test_box_match_counts_true_false_and_missed_detections() -> None:
    targets = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predictions = [
        {"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9},
        {"bbox_xyxy": [40, 40, 50, 50], "confidence": 0.8},
    ]

    assert _match_boxes(targets, predictions, threshold=0.4) == (1, 1, 1)


def test_select_threshold_maximizes_recall_subject_to_precision_constraint() -> None:
    metrics = {
        "0.2": {"box_precision_iou_0_50": 0.89, "box_recall_iou_0_50": 0.99, "count_mae": 1.0},
        "0.3": {"box_precision_iou_0_50": 0.92, "box_recall_iou_0_50": 0.80, "count_mae": 1.5},
        "0.4": {"box_precision_iou_0_50": 0.95, "box_recall_iou_0_50": 0.80, "count_mae": 1.0},
    }

    assert select_threshold(metrics, [0.2, 0.3, 0.4]) == 0.4


def test_select_threshold_fails_if_precision_constraint_is_not_met() -> None:
    metrics = {
        "0.2": {"box_precision_iou_0_50": 0.89, "box_recall_iou_0_50": 0.99, "count_mae": 1.0}
    }
    with pytest.raises(ValueError, match="Ningún umbral alcanza precisión"):
        select_threshold(metrics, [0.2])
