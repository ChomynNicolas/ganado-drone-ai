from __future__ import annotations

import json

import pytest
from PIL import Image

from scripts.evaluate_external_cattle import (
    average_precision,
    iou,
    load_samples,
    match_predictions,
    parse_supervisely_boxes,
)


def test_parse_supervisely_boxes_reads_cattle_rectangles() -> None:
    boxes = parse_supervisely_boxes(
        {
            "objects": [
                {
                    "classTitle": "cattle",
                    "geometryType": "rectangle",
                    "points": {"exterior": [[1, 2], [8, 10]]},
                },
                {
                    "classTitle": "sheep",
                    "geometryType": "rectangle",
                    "points": {"exterior": [[0, 0], [3, 3]]},
                },
            ]
        },
        width=10,
        height=12,
    )

    assert boxes == [[1.0, 2.0, 8.0, 10.0]]


def test_parse_supervisely_boxes_rejects_out_of_bounds_coordinates() -> None:
    with pytest.raises(ValueError, match="fuera de la imagen"):
        parse_supervisely_boxes(
            {
                "objects": [
                    {
                        "classTitle": "cattle",
                        "geometryType": "rectangle",
                        "points": {"exterior": [[2, 2], [12, 10]]},
                    }
                ]
            },
            width=10,
            height=12,
        )


def test_iou_and_fixed_threshold_matching() -> None:
    sample = {"ground_truth": [[0.0, 0.0, 10.0, 10.0]]}
    predictions = [
        [
            {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9},
            {"bbox_xyxy": [20.0, 20.0, 30.0, 30.0], "confidence": 0.8},
        ]
    ]

    metrics = match_predictions([sample], predictions, confidence=0.4, iou_threshold=0.5)

    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0


def test_average_precision_is_one_for_perfect_single_box() -> None:
    sample = {"ground_truth": [[0.0, 0.0, 10.0, 10.0]]}
    predictions = [[{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9}]]

    assert average_precision(
        [sample], predictions, confidence_floor=0.001, iou_threshold=0.5
    ) == pytest.approx(0.995)


def test_load_samples_refuses_incomplete_test_by_default(tmp_path) -> None:
    image_dir = tmp_path / "Dataset2" / "img"
    annotation_dir = tmp_path / "Dataset2" / "ann"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    Image.new("RGB", (20, 12), "green").save(image_dir / "DJI_0032.JPG")
    (annotation_dir / "DJI_0032.JPG.json").write_text(
        json.dumps(
            {
                "size": {"width": 20, "height": 12},
                "objects": [
                    {
                        "classTitle": "cattle",
                        "geometryType": "rectangle",
                        "points": {"exterior": [[1, 1], [5, 5]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Dataset 2 incompleto"):
        load_samples(tmp_path / "Dataset2", allow_partial=False)

    samples = load_samples(tmp_path / "Dataset2", allow_partial=True)
    assert len(samples) == 1
    assert len(samples[0]["ground_truth"]) == 1
