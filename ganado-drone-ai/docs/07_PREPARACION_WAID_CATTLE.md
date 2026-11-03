# Cambio 07 — Preparación de WAID para detectar ganado bovino

## Objetivo

Transformar de manera **reproducible** las anotaciones originales de WAID (`cattle=1` entre seis especies) a un dataset de detección de **una clase** (`cattle=0`). No se alteran los datos de origen. El split oficial solamente se conserva cuando no hay fotografías originales compartidas. Ante solapamiento, el conversor exige `--resplit-by-source-image` y reconstruye grupos sin fuga evidente de variantes.

## Precondiciones

1. WAID descargado en `data/raw/waid-repo/WAID/` siguiendo [`06_FUENTE_WAID.md`](06_FUENTE_WAID.md).
2. El auditor original permanece estricto. En WAID descargado el 03/11/2026 se hallaron 51 cajas degeneradas de `sheep=0`. El conversor solo admite esa excepción de manera explícita con `--allow-invalid-sheep-zero-boxes`; **cualquier otro error sigue siendo bloqueante**. También repite la verificación SHA-256.
3. Se detectaron 788 fotografías base con variantes en varios splits oficiales. **Obligatorio usar `--resplit-by-source-image`**: agrupa por nombre anterior a `.rf.`, asigna los grupos de forma reproducible y los estratifica por presencia de cattle. Los SHA-256 no detectan imágenes casi idénticas de distintos identificadores: queda pendiente inspección por vuelo/escena.
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
  --output data/processed/waid-cattle-clean-v1 `
  --max-train-positives 500 `
  --allow-invalid-sheep-zero-boxes `
  --resplit-by-source-image `
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
  --allow-invalid-sheep-zero-boxes `
  --resplit-by-source-image `
  --seed 42 `
  --source-revision $revision
```

**Los splits procesados se reconstruyen a nivel de fotografía original, no preservan los splits oficiales**. En ambas variantes, validación y test conservan todas las imágenes **reasignadas a sus splits y con un archivo de etiquetas existente**, incluidos los negativos de otras especies. Por defecto se aborta si hay imágenes sin TXT: corregí el origen o, solo tras inspeccionarlas, utilizá explícitamente `--skip-missing-labels` para excluirlas. No inventes etiquetas negativas de imágenes desconocidas.

## Estructura generada (ignorada por Git)

```text
data/processed/waid-cattle-clean-v1/
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
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/waid-cattle-clean-v1/labels --num-classes 1
```

Revisá el manifiesto:

```powershell
Get-Content data/processed/waid-cattle-clean-v1/manifest.json
Get-Content data/processed/waid-cattle-clean-v1/dataset.yaml
```

## Inspección visual (obligatoria antes de entrenar)

```powershell
.\.venv\Scripts\python.exe -m scripts.preview_cattle `
  --dataset data/processed/waid-cattle-clean-v1 `
  --split train `
  --count 12 `
  --output reports/generated/cattle-train-preview.jpg
```

Repetí con `--split valid` y `--split test`, cambiando `--output`. Abrí los JPG locales y verificá que las cajas coincidan con los animales y no con otras especies. El mosaico reduce imágenes para inspección rápida: abrí además **las originales** cuando los animales sean muy pequeños.

## Controles metodológicos

- **Leakage controlado por fotografía base:** agrupación por identificador previo a `.rf.` y hashes exactos. Todavía es necesario revisar escenas/vuelos; no afirmar independencia completa.
- **Negativos:** necesarios para medir falsos positivos; se seleccionan determinísticamente en train y se conservan todos los disponibles en val/test.
- **Sin transformaciones geométricas:** el conversor copia imágenes y solo recodifica IDs de clase; las coordenadas originales se preservan.
- **Test aislado:** no utilizar imágenes de `test` para escoger hiperparámetros o umbrales. Usar `valid` para ese propósito.
- **Limitación geográfica:** WAID no demuestra desempeño en estancias de Paraguay. Debe explicitarse en el informe y en la defensa.
- **Conteo:** suma de cajas por foto ≠ animales únicos cuando hay fotografías superpuestas.

## Fallos previsibles y acción

| Señal | Acción |
|---|---|
| `WAID no pasó auditoría` | Los 51 errores conocidos de sheep requieren `--allow-invalid-sheep-zero-boxes`; cualquier otro error exige investigación |
| `identificadores ... en varios splits` | Ejecutar `--resplit-by-source-image` para reagrupar las variantes de origen |
| `Hay ... imágenes sin TXT` | Revisar si están realmente sin anotaciones; evitar convertirlas silenciosamente en negativos |
| `El destino ya existe` | Utilizar otro nombre de salida para conservar reproducibilidad; nunca sobrescribir datos de otra ejecución |
| Poco espacio | Crear el subconjunto de entrenamiento; no eliminar test y valid procesados ni mezclar variantes entre splits |
| Cajas desplazadas en preview | Inspeccionar el TXT de origen y la imagen en resolución completa antes de entrenar |

## Criterio de aceptación del cambio

- Conversor repetible con semilla y parámetros registrados.
- Cada salida tiene `dataset.yaml`, `manifest.json`, imágenes y etiquetas de una sola clase.
- `validate_yolo_labels.py` no informa errores.
- Mosaicos de al menos train/valid/test revisados manualmente.
- No se versionaron datasets o previews protegidos por licencia.

## Investigación de la incidencia real

Ver [`08_INCIDENCIA_CALIDAD_WAID.md`](08_INCIDENCIA_CALIDAD_WAID.md), que contiene las observaciones obtenidas, decisiones de limpieza y criterios de aceptación.
