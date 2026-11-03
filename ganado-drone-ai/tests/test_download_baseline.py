"""Valida ubicación y hash del checkpoint sin descargar pesos reales."""

import hashlib
from pathlib import Path

import pytest

from scripts.download_baseline import download_checkpoint


def test_creates_checkpoint_and_reports_digest(tmp_path: Path):
    target = tmp_path / "models" / "yolo11n.pt"

    def fake_loader(name: str):
        assert name == "yolo11n.pt"
        Path(name).write_bytes(b"checkpoint-test")

    digest = download_checkpoint(target, loader=fake_loader)
    assert target.read_bytes() == b"checkpoint-test"
    assert digest == hashlib.sha256(b"checkpoint-test").hexdigest()


def test_preserves_existing_checkpoint(tmp_path: Path):
    target = tmp_path / "yolo11n.pt"
    target.write_bytes(b"existing")

    def must_not_download(_):
        raise AssertionError("should not be invoked")

    assert download_checkpoint(target, loader=must_not_download)


def test_rejects_different_checkpoint_name(tmp_path: Path):
    with pytest.raises(ValueError, match="yolo11n.pt"):
        download_checkpoint(tmp_path / "unknown.pt", loader=lambda _: None)
