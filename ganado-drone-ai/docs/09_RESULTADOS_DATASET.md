# Resultados reproducibles del dataset WAID — piloto cattle

**Origen:** https://github.com/xiaohuicui/WAID
**Revisión de la fuente:** `f6de7fb1fdd78cd794b31526dde0eb37fd321e38`
**Dataset derivado:** `data/processed/waid-cattle-clean-v1` (fuera de Git).
**Evidencia:** `manifest.json` y `split_group_audit.json` locales; verificados por script.

## Preparación

- Clase original: `cattle=1`; clase final: `cattle=0`.
- Política de splits: `grouped_source_image_70_20_10` (semilla `42`).
- Máximo de imágenes positivas en train: `500`.
- Ratio de negativos en train: `0.25`.
- Cajas degeneradas de `sheep` excluidas con autorización expresa: **51**.
- IDs base que originalmente cruzaban splits: **788**.
- IDs base que cruzan splits después de preparar: **0**.

## Salida comprobada

| Split | Imágenes | Positivas | Negativas | Cajas cattle | Grupos exportados |
|---|---:|---:|---:|---:|---:|
| `train` | 625 | 500 | 125 | 6608 | 624 |
| `valid` | 2887 | 933 | 1954 | 12798 | 2393 |
| `test` | 1434 | 466 | 968 | 6234 | 1196 |
| **Total** | **4946** | **1899** | **3047** | **25640** | — |

**Nota:** `train` es una muestra piloto. Los grupos asignados antes del submuestreo no equivalen a los grupos exportados.

## Limitaciones y validaciones pendientes

- Validación visual completa de cajas y escenas: **pendiente**.
- El agrupamiento por ID base no elimina imágenes diferentes del mismo vuelo o potrero.
- Licencia de redistribución de la fuente: **pendiente de verificar**.
- No hay métricas del detector todavía: estas son métricas de preparación de datos.
- El split `test` queda reservado para la evaluación final del modelo.

## Reproducción

La ruta del dataset y los resultados completos deben reconstruirse con `scripts/prepare_cattle_dataset.py` y `scripts/check_split_groups.py`; consultar `docs/07_PREPARACION_WAID_CATTLE.md`. Los pesos e imágenes no se incluyen en Git.
