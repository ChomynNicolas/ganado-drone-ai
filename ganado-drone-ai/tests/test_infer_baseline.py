"""Pruebas del CLI baseline sin importar modelos descargados ni PyTorch."""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.infer_baseline import (
    cow_class_id,
    cow_detections,
    image_paths,
    infer_images,
    select_device,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[1, 2, 10, 20], [3, 4, 9, 12]])
    conf = FakeTensor([0.85, 0.5])
    cls = FakeTensor([19, 18])


class FakePrediction:
    boxes = FakeBoxes()
    speed = {"inference": 12.5}

    def save(self, filename):
        Path(filename).write_bytes(b"fake-output")


class FakeModel:
    names = {18: "sheep", 19: "cow"}

    def predict(self, **kwargs):
        assert kwargs["classes"] == [19]
        return [FakePrediction()]


def test_cow_class_id_detects_coco_class():
    assert cow_class_id({18: "sheep", 19: "cow"}) == 19
    assert cow_class_id(["person", "cow"]) == 1


def test_cow_class_id_missing_is_error():
    with pytest.raises(ValueError, match="cow"):
        cow_class_id({0: "cattle"})


def test_cow_detections_filter_non_cow():
    detection = cow_detections(FakePrediction(), 19)
    assert len(detection) == 1
    assert detection[0]["bbox_xyxy"] == [1.0, 2.0, 10.0, 20.0]
    assert detection[0]["confidence"] == 0.85


def test_image_paths_reject_empty_and_zero_limit(tmp_path):
    (tmp_path / "notes.txt").write_text("not-image")
    with pytest.raises(ValueError, match="No se encontraron"):
        image_paths(tmp_path, 2)
    with pytest.raises(ValueError, match="mayor que cero"):
        image_paths(tmp_path, 0)


def test_image_paths_limits_sorted(tmp_path):
    (tmp_path / "b.jpg").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "c.jpg").touch()
    assert [x.name for x in image_paths(tmp_path, 2)] == ["a.png", "b.jpg"]


def test_select_device_requires_cuda_if_requested():
    assert select_device("auto", False) == "cpu"
    assert select_device("auto", True) == "0"
    with pytest.raises(ValueError, match="PyTorch"):
        select_device("cuda", False)


def test_infer_images_writes_consistent_json(tmp_path):
    image = tmp_path / "input.jpg"
    Image.new("RGB", (16, 12)).save(image)
    out = tmp_path / "out"
    report = infer_images(
        FakeModel(),
        [image],
        out,
        confidence=0.25,
        image_size_px=640,
        device="cpu",
        model_sha256="abc",
    )
    assert report["total_detections"] == 1
    assert report["images_processed"] == 1
    assert (out / "001_input.jpg").is_file()
    saved = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert saved["results"][0]["count"] == len(saved["results"][0]["detections"])


def test_infer_images_rejects_wrong_confidence(tmp_path):
    with pytest.raises(ValueError, match="conf"):
        infer_images(
            FakeModel(),
            [],
            tmp_path,
            confidence=0,
            image_size_px=640,
            device="cpu",
            model_sha256="abc",
        )
