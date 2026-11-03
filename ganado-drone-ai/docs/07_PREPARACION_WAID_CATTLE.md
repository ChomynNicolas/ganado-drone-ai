# Cambio 07 — Preparación de WAID para detectar ganado bovino

## Objetivo

Transformar de manera **reproducible** las anotaciones originales de WAID (`cattle=1` entre seis especies) a un dataset de detección de **una clase** (`cattle=0`). No se alteran los datos de origen ni se remezclan las particiones oficiales.

## Precondiciones

1. WAID descargado en `data/raw/waid-repo/WAID/` siguiendo [`06_FUENTE_WAID.md`](06_FUENTE_WAID.md).
2. El reporte de auditoría `reports/generated/waid_audit.json` no debe contener errores. El propio conversor repite la auditoría **incluyendo SHA-256 entre splits** antes de copiar nada.
3. Revisar manualmente anotaciones y escenas relacionadas. Los SHA-256 **no** detectan imágenes casi idénticas. Si existen vuelos o escenas superpuestas entre train/valid/test, separar por grupo antes de reportar resultados como independientes.
4. La fuente WAID pública no equivale a permiso de redistribución: no subir sus imágenes, anotaciones derivadas o previews a GitHub sin verificar licencia.

Instalar dependencia para las visualizaciones:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

## Generar un primer dataset de prueba (entrenamiento limitado)

Obtener el identificador de la revisión descargada (si la descarga fue vía Git):

```powershell
$revision = git -C data/raw/waid-repo rev-parse HEAD
```

Ejecutar (la ruta de salida **no debe existir**):

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_cattle_dataset `
  --source data/raw/waid-repo/WAID `
  --output data/processed/waid-cattle-pilot `
  --max-train-positives 500 `
  --train-negative-ratio 0.25 `
  --seed 42 `
  --source-revision $revision
```

Si descargaste un ZIP sin `.git`, eliminá el parámetro `--source-revision` y documentá la fecha y URL de descarga en `data/DATASET_CARD.md`.

`--max-train-positives 500` limita **solo imágenes de entrenamiento con ganado**. No se afirma que el subconjunto represente toda la variabilidad del dataset. `--train-negative-ratio 0.25` selecciona como máximo un negativo etiquetado cada cuatro positivos. Una imagen con otras especies y ninguna vaca es negativa válida para la tarea `cattle`.

Para una ejecución completa posterior, usá otra salida y omití `--max-train-positives`:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_cattle_dataset `
  --source data/raw/waid-repo/WAID `
  --output data/processed/waid-cattle-full `
  --train-negative-ratio 0.25 `
  --seed 42 `
  --source-revision $revision
```

**No se cambia train/valid/test del dataset de origen**. En ambas variantes, validación y test conservan todas las imágenes **con un archivo de etiquetas existente**, incluidos los negativos de otras especies. Por defecto se aborta si hay imágenes sin TXT: corregí el origen o, solo tras inspeccionarlas, utilizá explícitamente `--skip-missing-labels` para excluirlas. No inventes etiquetas negativas de imágenes desconocidas.

## Estructura generada (ignorada por Git)

```text
data/processed/waid-cattle-pilot/
├── dataset.yaml
├── manifest.json
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

Los TXT contienen solo `class_id=0`, con las coordenadas normalizadas originales. Cada imagen seleccionada tiene su TXT correspondiente, vacío para imágenes negativas verificadas. `dataset.yaml` tiene rutas absolutas de la máquina donde fue generado; **cada integrante debe regenerarlo localmente**. Los ejemplos del documento no son estadísticas reales del proyecto.

`manifest.json` documenta fuente, revisión (si fue indicada), clase de origen/destino, semilla, configuración y conteos por split. Conservá estos resultados localmente; en `data/DATASET_CARD.md` anotá los valores realmente obtenidos, sin subir datos de terceros.

## Validación del dataset procesado

```powershell
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/waid-cattle-pilot/labels --num-classes 1
```

Revisá el manifiesto:

```powershell
Get-Content data/processed/waid-cattle-pilot/manifest.json
Get-Content data/processed/waid-cattle-pilot/dataset.yaml
```

## Inspección visual (obligatoria antes de entrenar)

```powershell
.\.venv\Scripts\python.exe -m scripts.preview_cattle `
  --dataset data/processed/waid-cattle-pilot `
  --split train `
  --count 12 `
  --output reports/generated/cattle-train-preview.jpg
```

Repetí con `--split valid` y `--split test`, cambiando `--output`. Abrí los JPG locales y verificá que las cajas coincidan con los animales y no con otras especies. El mosaico reduce imágenes para inspección rápida: abrí además **las originales** cuando los animales sean muy pequeños.

## Controles metodológicos

- **No leakage evidente:** hashes exactos entre splits. Pendiente comprobar similitud visual entre escenas/vuelos; documentarlo como limitación.
- **Negativos:** necesarios para medir falsos positivos; se seleccionan determinísticamente en train y se conservan todos los disponibles en val/test.
- **Sin transformaciones geométricas:** el conversor copia imágenes y solo recodifica IDs de clase; las coordenadas originales se preservan.
- **Test aislado:** no utilizar imágenes de `test` para escoger hiperparámetros o umbrales. Usar `valid` para ese propósito.
- **Limitación geográfica:** WAID no demuestra desempeño en estancias de Paraguay. Debe explicitarse en el informe y en la defensa.
- **Conteo:** suma de cajas por foto ≠ animales únicos cuando hay fotografías superpuestas.

## Fallos previsibles y acción

| Señal | Acción |
|---|---|
| `WAID no pasó auditoría` | Volver al reporte de Cambio 06; revisar imágenes/etiquetas/duplicados antes de convertir |
| `Hay ... imágenes sin TXT` | Revisar si están realmente sin anotaciones; evitar convertirlas silenciosamente en negativos |
| `El destino ya existe` | Utilizar otro nombre de salida para conservar reproducibilidad; nunca sobrescribir datos de otra ejecución |
| Poco espacio | Crear el subconjunto de entrenamiento; no eliminar el test original ni mezclar splits |
| Cajas desplazadas en preview | Inspeccionar el TXT de origen y la imagen en resolución completa antes de entrenar |

## Criterio de aceptación del cambio

- Conversor repetible con semilla y parámetros registrados.
- Cada salida tiene `dataset.yaml`, `manifest.json`, imágenes y etiquetas de una sola clase.
- `validate_yolo_labels.py` no informa errores.
- Mosaicos de al menos train/valid/test revisados manualmente.
- No se versionaron datasets o previews protegidos por licencia.
