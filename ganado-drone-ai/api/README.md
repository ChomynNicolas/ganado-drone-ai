# API de inferencia

FastAPI expone `GET /health` y `POST /predict`. El modelo se carga una vez al
iniciar el proceso. La UI envía una imagen por petición y dibuja las cajas con
las coordenadas que devuelve la API.

## Ejecución local

Desde la raíz del proyecto:

```powershell
python -m pip install -r requirements-app.txt
$env:MODEL_WEIGHTS = 'models/waid-cattle-yolo11n-uav-mixed.pt'
$env:MODEL_DEVICE = 'auto'
$env:MODEL_CONFIDENCE = '0.40'
python -m uvicorn api.app.main:app --reload --port 8000
```

Si el checkpoint falta o no carga, `/health` mantiene el servicio visible con
`model_loaded: false` y `/predict` responde `503`. El endpoint admite JPG/JPEG/PNG,
limita cada imagen a 20 MiB y valida los bytes decodificados con Pillow.

- `GET /health`: disponibilidad del proceso y del detector.
- `POST /predict`: multipart con un campo `file`; incluye `count`, detecciones,
  dimensiones originales y latencia de inferencia.
- `GET /docs`: contrato OpenAPI generado por FastAPI.

Las cajas usan coordenadas `xyxy` en píxeles de la imagen original. El endpoint
devuelve conteos por imagen; un lote de fotos superpuestas puede repetir bovinos.

El checkpoint mixto se genera localmente y no está incluido en Git. Antes de
iniciar la API, confirma que `models/waid-cattle-yolo11n-uav-mixed.pt` exista.
