# Arquitectura técnica prevista — Ganado Drone AI

**Estado:** diseño aprobado para implementar. Este archivo **no** significa que los componentes ya funcionen.

## 1. Propósito y frontera del sistema

Entrada: fotografías aéreas de potreros (JPEG/PNG) suministradas por el usuario. Salida: bounding boxes, confidence y conteo de la clase `cattle` **por imagen**. El dashboard calcula la suma de detecciones del lote.

No existe integración con el dron, GPS, identificación individual, tracking, ni deduplicación entre imágenes en el MVP. Dos fotos de un mismo bovino pueden producir dos detecciones; nunca llamar «animales únicos» a la suma del lote.

## 2. Componentes

| Componente | Responsabilidad | Entrada → salida |
|---|---|---|
| `data` | registrar origen, limpiar y convertir etiquetas al esquema de una clase | dataset origen → imágenes + YOLO labels consistentes |
| `training` | baseline, ajuste, evaluación y exportación de checkpoint | dataset → pesos `.pt` + métricas |
| `api` | validar imágenes, cargar checkpoint, preprocesar, ejecutar inferencia y normalizar respuesta | multipart JPEG/PNG → JSON |
| `ui` | recibir un lote, llamar a la API por imagen, mostrar boxes y resumen | archivos del usuario → tabla + imágenes anotadas |
| `docker` | ejecutar API y UI offline con rutas y dependencias reproducibles | `docker compose up --build` → dos servicios locales |

La UI **no** importa directamente el modelo: debe llamar por HTTP a FastAPI para que exista una separación comprobable entre interfaz e inferencia.

## 3. Flujo de inferencia

1. La persona selecciona varias imágenes en el dashboard.
2. Streamlit valida la selección y envía **una imagen por petición** a `POST /predict` (procesamiento secuencial en el MVP).
3. FastAPI verifica tamaño, formato y que el archivo sea una imagen decodificable; rechaza entradas inválidas sin detener el lote completo.
4. Se ejecuta el detector cargado una sola vez durante el arranque o inicialización del proceso.
5. Se transforman las detecciones en coordenadas `xyxy` de píxeles de la imagen original, con confidence y etiqueta `cattle`.
6. Streamlit dibuja bounding boxes y muestra conteos por imagen; suma **detecciones** y presenta errores por fotografía si corresponde.

**Regla de integridad:** `count` debe igualar el número de objetos de `detections` retornados, sin crear valores simulados.

## 4. Contrato HTTP preliminar (a implementar en la fase API)

### `GET /health`

- Estado `200` si la aplicación responde; debe informar además `model_loaded` (booleano).
- `model_loaded=false` significa que la inferencia todavía no está disponible, no que el servicio esté listo para predecir.

Ejemplo documental (no corresponde a una ejecución):

```json
{"status": "ok", "model_loaded": false}
```

### `POST /predict`

- `Content-Type: multipart/form-data` con campo `file` (una fotografía).
- Si faltan pesos válidos, devolver `503` con mensaje explícito (no producir conteo ficticio).
- Extensiones objetivo: `.jpg`, `.jpeg` y `.png`; validar contenido realmente decodificable, no solo extensión ni MIME.
- Limitar tamaño de upload y cantidad de fotos del lote; valores exactos a fijar después de pruebas de memoria.

Respuesta de ejemplo **solo de esquema** (coordenadas ficticias para ilustrar formato; no son predicciones):

```json
{
  "image_name": "sample.jpg",
  "width": 1920,
  "height": 1080,
  "count": 1,
  "detections": [
    {"class_name": "cattle", "confidence": 0.91, "bbox_xyxy": [120, 240, 180, 305]}
  ],
  "inference_ms": 43.2
}
```

`inference_ms`: tiempo de predicción del detector por imagen (no incluye upload/rendering); definir medición de warmup y hardware al implementar benchmarks. No se fijan objetivos de latencia sin medirlos.

## 5. Entrenamiento y validación

- Transformar el dataset a una única clase `cattle` **después** de inspeccionar su mapeo real.
- Particionar antes del entrenamiento y controlar fuga de datos entre escenas/vuelos similares; la partición publicada por el dataset no se asume automáticamente segura.
- Comparar baseline y modelo ajustado mediante el mismo protocolo; usar validation para iteraciones y test para la evaluación final.
- Guardar configuración de experimentos, versiones, semillas, logs, métricas e imágenes de errores representativos.
- Priorizar GPU local GTX 1650 4 GB con modelo pequeño y batch adaptado a memoria; usar Colab gratuito si no alcanza.

## 6. Estructura objetivo (todavía NO creada)

```text
api/app/            # aplicación FastAPI y servicio del detector
ui/                 # dashboard Streamlit
scripts/            # adquisición/preparación, train y evaluate
data/               # dataset local (ignorado por Git) + datasheet versionado
configs/            # configuración de dataset y experimentos
models/             # checkpoints (ignorados por Git)
tests/              # pruebas unitarias, contratos y smoke tests
docker/             # Dockerfiles
reports/            # tablas/figuras verificables del informe
docs/               # arquitectura, ADR, procesos y registros
```

Se crearán las carpetas **a medida que tengan contenido real**. No agregar placeholders innecesarios para aparentar avance.

## 7. Gates de integración

1. **Datos:** al menos un lote de imágenes y etiquetas reales auditadas, con distribución de clases y licencia documentadas.
2. **Modelo:** pesos reales cargables; inferencia que dibuja cajas y reporta valores verificables.
3. **API:** `/health`, `/predict` y pruebas de manejo de archivos válidos, inválidos y pesos ausentes.
4. **UI:** procesa dos o más imágenes con sus conteos y total de detecciones.
5. **Entrega:** despliegue y demo offline repetibles; reporte IEEE con métricas medidas, no supuestas.

## 8. Riesgos y controles

- **Animales pequeños:** probar resolución de entrenamiento y, si corresponde, tiling; registrar impacto en mAP/recall y duplicados en bordes de tiles.
- **Solapamiento de fotos:** sin deduplicación; explicar límite en UI e informe.
- **VRAM:** usar batch pequeño / disminuir resolución solo si se evalúa la pérdida de recall, o cambiar a Colab.
- **Carga maliciosa o corrupta:** límites y validación estricta de archivos; no guardar nombres de usuario como rutas del servidor.
- **Dependencia de Internet:** inferencia y demo offline, con checkpoints y ejemplos disponibles localmente.

## Referencias de implementación

- <https://docs.ultralytics.com/models/yolo11/>
- <https://fastapi.tiangolo.com/tutorial/request-files/>
- <https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader>
