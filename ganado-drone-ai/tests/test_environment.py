"""Pruebas del verificador de entorno; no requieren librerias de ML."""

from pathlib import Path

import pytest

from scripts.check_environment import (
    read_expected_python,
    read_pinned_requirements,
    validate_environment,
)


def test_read_expected_python(tmp_path: Path) -> None:
    version_file = tmp_path / ".python-version"
    version_file.write_text("3.11\n", encoding="utf-8")
    assert read_expected_python(version_file) == (3, 11)


def test_read_pinned_requirements(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements-dev.txt"
    requirements.write_text("# tools\npytest==9.2.0\nruff==0.17.3\n", encoding="utf-8")
    assert read_pinned_requirements(requirements) == {"pytest": "9.2.0", "ruff": "0.17.3"}


def test_reject_unpinned_requirements(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements-dev.txt"
    requirements.write_text("pytest>=9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sin version exacta"):
        read_pinned_requirements(requirements)


def test_valid_environment() -> None:
    errors = validate_environment(
        (3, 11),
        (3, 11),
        {"pytest": "9.2.0", "ruff": "0.17.3"},
        {"pytest": "9.2.0", "ruff": "0.17.3"},
    )
    assert errors == []


def test_reject_wrong_python_version() -> None:
    errors = validate_environment((3, 12), (3, 11), {}, {})
    assert errors == ["Python 3.11 requerido; detectado 3.12"]


def test_missing_or_mismatched_dependencies() -> None:
    errors = validate_environment(
        (3, 11),
        (3, 11),
        {"pytest": "9.2.0", "ruff": "0.17.3"},
        {"pytest": "9.0.0", "ruff": None},
    )
    assert "pytest: version esperada 9.2.0; instalada 9.0.0" in errors
    assert "Falta ruff==0.17.3" in errors
