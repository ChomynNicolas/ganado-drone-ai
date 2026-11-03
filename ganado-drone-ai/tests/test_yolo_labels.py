"""Pruebas unitarias del validador de etiquetas YOLO."""

from pathlib import Path

import pytest

from scripts.validate_yolo_labels import (
    main,
    validate_label_file,
    validate_labels,
    validate_yolo_line,
)


def test_valid_single_class_box() -> None:
    assert validate_yolo_line("0 0.5 0.5 0.3 0.2") == (0, 0.5, 0.5, 0.3, 0.2)


@pytest.mark.parametrize(
    ("line", "error"),
    [
        ("0 0.5 0.5 0.2", "5 columnas"),
        ("1 0.5 0.5 0.2 0.1", "fuera de rango"),
        ("0.0 0.5 0.5 0.2 0.1", "entero"),
        ("0 1.2 0.5 0.2 0.1", "centro"),
        ("0 0.5 0.5 0.0 0.1", "width"),
        ("0 0.9 0.5 0.5 0.1", "límites"),
        ("0 NaN 0.5 0.2 0.1", "finitas"),
        ("0 0.5 inf 0.2 0.1", "finitas"),
        ("0 0.5 abc 0.2 0.1", "numéricas"),
        ("-1 0.5 0.5 0.2 0.1", "entero"),
    ],
)
def test_reject_malformed_boxes(line: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_yolo_line(line)


def test_accept_border_box_and_multiple_classes() -> None:
    assert validate_yolo_line("1 0.5 0.5 1.0 1.0", num_classes=2)[0] == 1


def test_empty_label_file_is_valid_negative_image(tmp_path: Path) -> None:
    label = tmp_path / "negative.txt"
    label.write_text("", encoding="utf-8")
    assert validate_label_file(label) == (0, [])


def test_duplicate_labels_report_line_number(tmp_path: Path) -> None:
    label = tmp_path / "cow.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    boxes, errors = validate_label_file(label)
    assert boxes == 1
    assert len(errors) == 1
    assert "cow.txt:2" in errors[0]
    assert "duplicada" in errors[0]


def test_recursive_directory_summary(tmp_path: Path) -> None:
    nested = tmp_path / "train"
    nested.mkdir()
    (nested / "a.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    (nested / "b.txt").write_text("", encoding="utf-8")
    result = validate_labels(tmp_path)
    assert (result.files, result.boxes, result.errors) == (2, 1, ())


def test_non_utf8_is_reported(tmp_path: Path) -> None:
    label = tmp_path / "broken.txt"
    label.write_bytes(b"\xff\xfe")
    boxes, errors = validate_label_file(label)
    assert boxes == 0
    assert len(errors) == 1
    assert "UTF-8" in errors[0]


def test_missing_path_is_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no existe"):
        validate_labels(tmp_path / "missing")


def test_empty_directory_is_not_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no se encontraron"):
        validate_labels(tmp_path)


def test_cli_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    label = tmp_path / "labels.txt"
    label.write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    assert main(["--labels", str(label)]) == 0
    assert "Cajas válidas: 1" in capsys.readouterr().out
    label.write_text("0 2.0 0.5 0.2 0.1\n", encoding="utf-8")
    assert main(["--labels", str(label)]) == 1
    assert "Errores encontrados: 1" in capsys.readouterr().out
