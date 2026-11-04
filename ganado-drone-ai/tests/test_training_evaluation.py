from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_yolo import label_count, mean_absolute_error, read_dataset_root
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
