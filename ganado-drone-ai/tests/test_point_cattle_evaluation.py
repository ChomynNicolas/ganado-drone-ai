from __future__ import annotations

import pytest
from PIL import Image

from scripts.evaluate_point_cattle import load_point_dataset, match_points, read_points


def test_read_points_parses_xy_and_rejects_out_of_bounds(tmp_path) -> None:
    annotation = tmp_path / "points.txt"
    annotation.write_text("2,3\n8,9\n", encoding="utf-8")
    assert read_points(annotation, width=10, height=10) == [[2.0, 3.0], [8.0, 9.0]]

    annotation.write_text("10,9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fuera de límites"):
        read_points(annotation, width=10, height=10)


def test_match_points_counts_hits_false_boxes_and_unmatched_points() -> None:
    points = [[5.0, 5.0], [25.0, 25.0]]
    detections = [
        {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9},
        {"bbox_xyxy": [30.0, 30.0, 40.0, 40.0], "confidence": 0.8},
    ]

    assert match_points(points, detections) == {"tp": 1, "fp": 1, "fn": 1}


def test_load_point_dataset_requires_full_set_unless_partial_flag(tmp_path) -> None:
    image = tmp_path / "one.png"
    Image.new("RGBA", (20, 12), "green").save(image)
    (tmp_path / "one.png.mask.0.txt").write_text("4,5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incompleto"):
        load_point_dataset(tmp_path)

    samples = load_point_dataset(tmp_path, allow_partial=True)
    assert len(samples) == 1
    assert samples[0]["points"] == [[4.0, 5.0]]
