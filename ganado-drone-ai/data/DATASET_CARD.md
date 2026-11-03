# Datasheet — WAID → cattle (una clase)

**Estado:** WAID descargado y auditado localmente por el integrante; conversión limpia aún pendiente de ejecución y comprobación. El reporte de auditoría contiene errores explícitos identificados y documentados.

| Campo | Registro |
|---|---|
| Dataset fuente | WAID — Wildlife Aerial Images from Drone |
| Autores | Mou, Liu, Zhu y Cui (2023) |
| Artículo | https://doi.org/10.3390/app131810397 |
| Repositorio oficial | https://github.com/xiaohuicui/WAID |
| Licencia del repositorio fuente | **Por verificar antes de redistribución** |
| Clases fuente | `0 sheep`, `1 cattle`, `2 seal`, `3 camelus`, `4 kiang`, `5 zebra` |
| Clase procesada | `0 cattle` |
| Objetivo | Detección y conteo de bovinos en fotografías aéreas |
| Imágenes paraguayas | No acreditadas; evaluar dominio de destino por separado |
| Formato | YOLO: `class_id x_center y_center width height` |
| Splits | Fuente `train`, `valid`, `test`; procesados con agrupación por fotografía base debido a solapamiento entre originales |
| Preparación | [`docs/07_PREPARACION_WAID_CATTLE.md`](../docs/07_PREPARACION_WAID_CATTLE.md) |
| Código | `scripts/prepare_cattle_dataset.py`, `scripts/preview_cattle.py` |

## Versión utilizada — ejecución verificada (2026-11-03)

WAID fue auditado y el conjunto piloto de una clase fue preparado localmente. La evidencia de la ejecución se genera directamente desde `manifest.json` y `split_group_audit.json` mediante `scripts/record_dataset_results.py`. No se incluyen fotografías ni resultados que no hayan sido ejecutados.

**Resumen comunicado por el equipo:** 14.366 imágenes originales; 51 cajas degeneradas de `sheep` excluidas; 788 IDs base cruzaban particiones originalmente. Dataset limpio: 625 imágenes `train`, 2.887 `valid`, 1.434 `test` (4.946 total), 25.640 cajas bovinas, 0 grupos compartidos y 0 errores en etiquetas procesadas.

- Evidencia versionada: [`docs/09_RESULTADOS_DATASET.md`](../docs/09_RESULTADOS_DATASET.md), generada en la computadora con los manifiestos reales.
- Revisión exacta de WAID y parámetros: documento generado, no rellenar con un hash inventado.
- Inspección visual de las imágenes y control de escenas del mismo vuelo: **pendientes**.
- La muestra de `train` de 625 imágenes es un **piloto**, no un entrenamiento final.

## Política de preparación

1. Conservar el dataset original sin modificaciones en `data/raw/`.
2. Verificar el orden de las seis clases antes de transformar.
3. Eliminar las anotaciones de clases no bovinas y reindexar `cattle=1` a `cattle=0`.
4. Reasignar grupos de fotografías originales con `.rf.` de manera reproducible para evitar variantes entre train/valid/test; después conservar todas las imágenes etiquetadas de `valid/test`, incluidas negativas.
5. No inferir que un archivo sin etiquetas es un negativo; el script aborta salvo exclusión explícita.
6. Registrar semilla, muestra y conteos en `manifest.json`.
7. Reservar `test` únicamente para la evaluación final; no usarlo para elegir hiperparámetros.
8. Excluir bajo opt-in solamente etiquetas degeneradas de `sheep=0` (ancho o alto cero), registrar cada exclusión; detectar duplicados exactos y documentar la limitación por fotografías distintas del mismo vuelo.

**Advertencia:** la suma de detecciones en fotos superpuestas puede duplicar animales; un detector por imagen no identifica individuos únicos entre fotografías.
