# Datasheet — WAID → cattle (una clase)

**Estado:** WAID seleccionado como fuente de trabajo; **no se confirma descarga, auditoría ni conversión reales en este archivo**. Los datos locales deben completarse al ejecutar los scripts.

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
| Splits | Fuente `train`, `valid`, `test`, conservados por el conversor |
| Preparación | [`docs/07_PREPARACION_WAID_CATTLE.md`](../docs/07_PREPARACION_WAID_CATTLE.md) |
| Código | `scripts/prepare_cattle_dataset.py`, `scripts/preview_cattle.py` |

## Versión utilizada — completar luego de ejecución real

- Fecha de descarga: **pendiente**
- Hash del commit WAID o URL exacta del ZIP: **pendiente**
- Auditoría `--hash-check`: **pendiente de registrar resultado**
- Conversión (ruta local y parámetros): **pendiente**
- Train: imágenes / cajas cattle / negativos = **pendiente**
- Valid: imágenes / cajas cattle / negativos = **pendiente**
- Test: imágenes / cajas cattle / negativos = **pendiente**
- Errores y advertencias del auditor: **pendiente**
- Revisión visual de cajas en los tres splits: **pendiente**
- Evaluación de escenas o vuelos similares entre splits: **pendiente**

## Política de preparación

1. Conservar el dataset original sin modificaciones en `data/raw/`.
2. Verificar el orden de las seis clases antes de transformar.
3. Eliminar las anotaciones de clases no bovinas y reindexar `cattle=1` a `cattle=0`.
4. Conservar todas las imágenes etiquetadas de `valid/test`, incluidas las negativas; permitir subconjunto determinista en `train` para experimentar.
5. No inferir que un archivo sin etiquetas es un negativo; el script aborta salvo exclusión explícita.
6. Registrar semilla, muestra y conteos en `manifest.json`.
7. Reservar `test` únicamente para la evaluación final; no usarlo para elegir hiperparámetros.
8. Detectar duplicados exactos y documentar la limitación por fotografías casi idénticas o del mismo vuelo.

**Advertencia:** la suma de detecciones en fotos superpuestas puede duplicar animales; un detector por imagen no identifica individuos únicos entre fotografías.
