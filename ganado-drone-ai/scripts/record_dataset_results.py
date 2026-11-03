"""Genera evidencia versionable a partir del manifiesto local de WAID.

No guarda rutas absolutas, datos personales ni imágenes originales en Git.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SPLITS = ("train", "valid", "test")


def build_report(manifest: dict, groups: dict) -> str:
    """Valida evidencia local y produce un resumen Markdown determinista."""
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
        raise ValueError("El manifiesto no contiene una revisión Git válida de WAID")
    if groups.get("overlap_count") != 0 or groups.get("ok") is not True:
        raise ValueError("La auditoría de grupos no pasó: no se puede registrar como válida")

    summaries = manifest.get("splits")
    if not isinstance(summaries, dict):
        raise ValueError("El manifiesto no contiene splits")

    table_rows = []
    totals = {"images": 0, "positive_images": 0, "negative_images": 0, "cattle_boxes": 0}
    for split in SPLITS:
        info = summaries.get(split)
        if not isinstance(info, dict):
            raise ValueError(f"Falta el split {split} en el manifiesto")
        for key in totals:
            if not isinstance(info.get(key), int) or info[key] < 0:
                raise ValueError(f"Campo inválido en {split}: {key}")
            totals[key] += info[key]
        if info["images"] != info["positive_images"] + info["negative_images"]:
            raise ValueError(f"Conteos positivos/negativos inconsistentes: {split}")
        if groups.get("images_per_split", {}).get(split) != info["images"]:
            raise ValueError(f"El reporte de grupos no coincide con el manifiesto: {split}")
        n_groups = groups.get("groups_per_split", {}).get(split)
        if not isinstance(n_groups, int) or n_groups < 1:
            raise ValueError(f"Cantidad de grupos inválida: {split}")
        table_rows.append(
            f"| `{split}` | {info['images']} | {info['positive_images']} | "
            f"{info['negative_images']} | {info['cattle_boxes']} | {n_groups} |"
        )

    params = manifest.get("parameters", {})
    if manifest.get("split_policy") != "grouped_source_image_70_20_10":
        raise ValueError("La salida no utilizó partición agrupada por origen")
    for key in ("seed", "train_negative_ratio", "max_train_positives"):
        if key not in params:
            raise ValueError(f"Falta parámetro de reproducibilidad: {key}")
    if not params.get("allow_invalid_sheep_zero_boxes"):
        raise ValueError("No se registró la política explícita de cajas degeneradas")

    return "\n".join(
        [
            "# Resultados reproducibles del dataset WAID — piloto cattle",
            "",
            "**Origen:** https://github.com/xiaohuicui/WAID",
            f"**Revisión de la fuente:** `{revision}`",
            "**Dataset derivado:** `data/processed/waid-cattle-clean-v1` (fuera de Git).",
            "**Evidencia:** `manifest.json` y `split_group_audit.json` locales; verificados por script.",
            "",
            "## Preparación",
            "",
            "- Clase original: `cattle=1`; clase final: `cattle=0`.",
            f"- Política de splits: `{manifest['split_policy']}` (semilla `{params['seed']}`).",
            f"- Máximo de imágenes positivas en train: `{params['max_train_positives']}`.",
            f"- Ratio de negativos en train: `{params['train_negative_ratio']}`.",
            "- Cajas degeneradas de `sheep` excluidas con autorización expresa: "
            f"**{manifest['excluded_invalid_sheep_zero_boxes_count']}**.",
            "- IDs base que originalmente cruzaban splits: "
            f"**{manifest['original_source_ids_crossing_splits']}**.",
            f"- IDs base que cruzan splits después de preparar: **{groups['overlap_count']}**.",
            "",
            "## Salida comprobada",
            "",
            "| Split | Imágenes | Positivas | Negativas | Cajas cattle | Grupos exportados |",
            "|---|---:|---:|---:|---:|---:|",
            *table_rows,
            f"| **Total** | **{totals['images']}** | **{totals['positive_images']}** | "
            f"**{totals['negative_images']}** | **{totals['cattle_boxes']}** | — |",
            "",
            "**Nota:** `train` es una muestra piloto. Los grupos asignados antes del "
            "submuestreo no equivalen a los grupos exportados.",
            "",
            "## Limitaciones y validaciones pendientes",
            "",
            "- Validación visual completa de cajas y escenas: **pendiente**.",
            "- El agrupamiento por ID base no elimina imágenes diferentes del mismo vuelo o potrero.",
            "- Licencia de redistribución de la fuente: **pendiente de verificar**.",
            "- No hay métricas del detector todavía: estas son métricas de preparación de datos.",
            "- El split `test` queda reservado para la evaluación final del modelo.",
            "",
            "## Reproducción",
            "",
            "La ruta del dataset y los resultados completos deben reconstruirse con "
            "`scripts/prepare_cattle_dataset.py` y `scripts/check_split_groups.py`; "
            "consultar `docs/07_PREPARACION_WAID_CATTLE.md`. Los pesos e imágenes no se "
            "incluyen en Git.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registrar auditoría WAID como Markdown")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/09_RESULTADOS_DATASET.md"))
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        groups = json.loads(args.groups.read_text(encoding="utf-8"))
        report = build_report(manifest, groups)
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"No se pudo registrar evidencia: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Documento generado: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
