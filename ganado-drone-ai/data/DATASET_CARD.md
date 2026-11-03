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

## Versión utilizada — completar luego de ejecución real

- Fecha de descarga: **pendiente**
- Hash del commit WAID o URL exacta del ZIP: **pendiente**
- Auditoría `--hash-check`: **ejecutada; 14.366 imágenes, 63.090 cajas válidas cattle; 51 errores `sheep=0`**
- Conversión (ruta local y parámetros): **pendiente**
- Train: imágenes / cajas cattle / negativos = **pendiente**
- Valid: imágenes / cajas cattle / negativos = **pendiente**
- Test: imágenes / cajas cattle / negativos = **pendiente**
- Errores y advertencias del auditor: **51 errores `width/height` de sheep y 2 advertencias**; se detectaron **788 IDs base en varios splits** mediante comprobación adicional
- Revisión visual de cajas en los tres splits: **pendiente**
- Evaluación de escenas o vuelos similares entre splits: **pendiente**

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
