# Primer baseline YOLO11n — detección COCO `cow`

**Estado:** código para ejecutar y probar localmente, **no** fine-tuning de ganado ni métricas de test.

## Intención

Verificar que PyTorch, Ultralytics, el checkpoint y el hardware ejecutan detecciones reales con cajas, conteo por fotografía y registro reproducible. YOLO11n está preentrenado en COCO y reconoce `cow` (ID 19), pero no necesariamente ganado diminuto visto desde drones. Las clases WAID (`cattle=0` después de convertir) pertenecen a otro esquema: el script identifica por nombre la clase COCO `cow`.

## Instalación en Windows (Python 3.11)

Desde el directorio raíz del proyecto, utilizar el intérprete de `.venv`. Antes de instalar PyTorch CUDA, verificar el driver con `nvidia-smi`.

Opción A — GTX 1650 con build CUDA 12.6 compatible con el driver:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Opción B — CPU, si falla la ruta CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

En caso de incompatibilidad de driver, revisar https://pytorch.org/get-started/previous-versions/ y el selector oficial; no afirmar que CUDA funciona hasta comprobarlo. Un cambio de versión obliga a registrar las dependencias efectivas del experimento.

```powershell
.\.venv\Scripts\python.exe -c "import torch, ultralytics; print('torch:',torch.__version__,'ultralytics:',ultralytics.__version__,'CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Pesos del modelo

```powershell
.\.venv\Scripts\python.exe -m scripts.download_baseline
```

Descarga el checkpoint oficial `yolo11n.pt` mediante Ultralytics y lo almacena en `models/`. Imprime SHA-256. No se versiona el archivo `.pt`, ni se descargan modelos durante la ejecución de las pruebas unitarias.

## Inferencia de prueba (solo imágenes de `train`)

**No se recomienda usar test para esta primera prueba ni ajustar umbrales mirando test.** Para seleccionar una imagen con cajas reales:

```powershell
$labelsDir = 'data/processed/waid-cattle-clean-v1/labels/train'
$imagesDir = 'data/processed/waid-cattle-clean-v1/images/train'
$label = Get-ChildItem $labelsDir -Filter '*.txt' | Where-Object { $_.Length -gt 0 } | Select-Object -First 1
if (-not $label) { throw 'No hay etiquetas positivas en train' }
$image = Get-ChildItem $imagesDir -File | Where-Object { $_.BaseName -eq $label.BaseName } | Select-Object -First 1
if (-not $image) { throw 'No se encontró la imagen de la etiqueta seleccionada' }
.\.venv\Scripts\python.exe -m scripts.infer_baseline --source $image.FullName --imgsz 640 --conf 0.25 --device auto
```

Para un lote pequeño (incluye imágenes negativas en caso de existir):

```powershell
.\.venv\Scripts\python.exe -m scripts.infer_baseline --source data/processed/waid-cattle-clean-v1/images/train --limit 5 --imgsz 640 --device auto
```

Salida: `reports/generated/baseline/predictions.json` y JPEGs anotados. El JSON incluye SHA-256 del modelo, nombres de imágenes, conteos, coordenadas, confianza, tiempo de inferencia reportado por el modelo y tiempo medido del predict. Son conteos **por imagen**; sumarlos no identifica animales únicos. El script no calcula mAP ni MAE.

## Pruebas y aprobación

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check scripts tests
.\.venv\Scripts\python.exe -m ruff format --check scripts tests
```

Para cerrar esta tarea: 1) al menos una fotografía procesada con inferencia real, 2) JSON y anotación visual válidos, 3) dispositivo y versiones registrados, 4) sin archivos de pesos o fotografías en Git. **Cero detecciones también es un resultado real**, no debe maquillarse; puede indicar limitaciones del baseline preentrenado.

## Siguiente fase

Fine-tuning sobre `train`, selección por `valid`, evaluación final con `test`, métricas mAP y error de conteo. No usar el peso `yolo11n.pt` directamente como supuesta solución final.
