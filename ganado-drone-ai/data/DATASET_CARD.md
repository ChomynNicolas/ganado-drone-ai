# Datasheet — WAID (fuente candidata seleccionada para auditoría)

**Estado:** fuente identificada y formato verificado en el repositorio oficial; **descarga y validación local pendientes**.

| Campo | Registro |
|---|---|
| Dataset | WAID — Wildlife Aerial Images from Drone |
| Artículo | Mou, Liu, Zhu y Cui, *Applied Sciences* 13(18), 10397 (2023) |
| DOI | https://doi.org/10.3390/app131810397 |
| Fuente original | https://github.com/xiaohuicui/WAID |
| Estructura observada | `WAID/classes.txt`, `WAID/images/{train,valid,test}`, `WAID/labels/{train,valid,test}` |
| Etiquetas | YOLO: `class_id x_center y_center width height` |
| Clases originales | `0 sheep`, `1 cattle`, `2 seal`, `3 camelus`, `4 kiang`, `5 zebra` |
| Clase del futuro modelo | `0 cattle` tras filtrar/reindexar en **Cambio 07** |
| Resolución típica | 640×640 según catálogo secundario; comprobar localmente |
| Cantidad de imágenes/anotaciones | **Pendiente de medir con auditoría local** |
| Splits | Originales train/valid/test; no mezclar antes de inspeccionar leakage |
| Licencia | **No confirmada para el repositorio original**; fuente pública no implica permiso de redistribución |
| Procedencia paraguaya | No acreditada; limitación de transferencia de dominio |
| Fecha/commit de descarga | **Pendiente de registrar después de descargar** |
| SHA-256 / duplicados | **Pendiente**; correr auditoría con `--hash-check` |

## Adquisición local y reproducibilidad

Consultar [`docs/06_FUENTE_WAID.md`](../docs/06_FUENTE_WAID.md). El equipo debe registrar el hash del commit fuente:

```powershell
git -C data/raw/waid-repo rev-parse HEAD
```

La descarga se guarda en `data/raw/waid-repo/`, ignorado por Git. Versionar la ficha y las herramientas, no imágenes ni etiquetas de terceros.

## Auditoría pendiente

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
```

La salida debe conservarse localmente para contrastar cantidades reales con las publicadas. Revisar errores, advertencias y muestras visuales. **No afirmar todavía que se dispone de un dataset apto para entrenamiento.**

## Riesgos de metodología

- Detección de duplicados por SHA-256 solo verifica bytes idénticos; no escenas muy similares o versiones aumentadas de la misma imagen.
- No hay evidencia de separación por vuelo/escena en el repositorio de origen; se investigará antes de confiar en métricas de test.
- Imágenes sin etiquetas pueden ser verdaderos negativos o anotaciones faltantes; el auditor avisa pero no decide.
- El conteo acumulado de cajas sobre imágenes superpuestas **no equivale al número de animales únicos**.
