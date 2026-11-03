"""Verifica la version de Python y las herramientas fijadas para el desarrollo."""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_expected_python(path: Path) -> tuple[int, int]:
    """Lee la version mayor/menor exigida por .python-version."""
    value = path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(".python-version debe contener MAYOR.MENOR, ejemplo: 3.11")
    return int(match.group(1)), int(match.group(2))


def read_pinned_requirements(path: Path) -> dict[str, str]:
    """Lee dependencias de desarrollo con version exacta (name==version)."""
    packages: dict[str, str] = {}
    for number, original_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = original_line.partition("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*)==([A-Za-z0-9_.+-]+)", line)
        if match is None:
            raise ValueError(f"{path.name}:{number}: dependencia sin version exacta: {line}")
        name, version = match.groups()
        normalized_name = name.lower().replace("_", "-")
        if normalized_name in packages:
            raise ValueError(f"{path.name}:{number}: dependencia duplicada: {name}")
        packages[normalized_name] = version
    return packages


def inspect_versions(package_names: list[str]) -> dict[str, str | None]:
    """Obtiene versiones realmente instaladas en el interprete activo."""
    installed: dict[str, str | None] = {}
    for name in package_names:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    return installed


def validate_environment(
    current_python: tuple[int, int],
    expected_python: tuple[int, int],
    expected_packages: Mapping[str, str],
    installed_packages: Mapping[str, str | None],
) -> list[str]:
    """Devuelve problemas detectados; una lista vacia significa entorno valido."""
    errors: list[str] = []
    if current_python != expected_python:
        errors.append(
            f"Python {expected_python[0]}.{expected_python[1]} requerido; "
            f"detectado {current_python[0]}.{current_python[1]}"
        )

    for name, expected_version in expected_packages.items():
        installed_version = installed_packages.get(name)
        if installed_version is None:
            errors.append(f"Falta {name}=={expected_version}")
        elif installed_version != expected_version:
            errors.append(
                f"{name}: version esperada {expected_version}; instalada {installed_version}"
            )
    return errors


def main() -> int:
    expected_python = read_expected_python(PROJECT_ROOT / ".python-version")
    requirements = read_pinned_requirements(PROJECT_ROOT / "requirements-dev.txt")
    current_python = (sys.version_info.major, sys.version_info.minor)
    installed = inspect_versions(list(requirements))
    errors = validate_environment(current_python, expected_python, requirements, installed)

    print(f"Python activo: {sys.version.split()[0]}")
    for name, expected in requirements.items():
        print(f"{name}: esperado={expected}, instalado={installed[name] or 'ausente'}")

    if errors:
        print("\nEntorno no valido:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nEntorno de desarrollo verificado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
