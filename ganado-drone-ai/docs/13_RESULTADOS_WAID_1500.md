# Tercer entrenamiento YOLO11n: 1.500 imágenes WAID

## Objetivo

Medir el efecto de ampliar el conjunto de entrenamiento frente al piloto de 625
imágenes, manteniendo modelo inicial, semilla, hiperparámetros y particiones de
validación/test.

## Datos y controles

- Fuente WAID: revisión `f6de7fb1fdd78cd794b31526dde0eb37fd321e38`.
- Entrenamiento: 1.200 imágenes positivas y 300 negativas, 1.500 imágenes en
  total; 16.175 cajas cattle.
- Validación: 2.887 imágenes y 12.798 cajas cattle.
- Test: 1.434 imágenes y 6.234 cajas cattle.
- Las imágenes y etiquetas de validación/test son idénticas byte por byte a las
  del piloto. El nuevo train contiene las 625 imágenes del piloto.
- Se validaron 5.821 etiquetas, sin errores, y no hay grupos de fotografía
  fuente compartidos entre splits.
- Se inspeccionó visualmente una muestra de 9 imágenes por split.

El dataset generado y los mosaicos permanecen locales; no forman parte de este
commit.

## Configuración

- Modelo inicial: YOLO11n preentrenado en COCO (`models/yolo11n.pt`).
- 50 épocas, tamaño de imagen 640, batch 8, paciencia 10.
- Semilla 42, workers 0, dispositivo automático.
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU.
- Entorno: Python 3.11.9, PyTorch 2.9.1+cu126, Ultralytics 8.4.198.
- Mejor checkpoint: época 50.

## Resultados en validación

| Métrica | Resultado |
| --- | ---: |
| Precision | 0.948680 |
| Recall | 0.917240 |
| mAP@0.5 | 0.960150 |
| mAP@0.5:0.95 | 0.622430 |

El umbral de confianza para conteo se seleccionó en validación: `0.35`.

## Comparación en test

Ambos modelos se evaluaron sobre las mismas 1.434 imágenes de test.

| Métrica | Piloto 625 | Train 1.500 | Diferencia |
| --- | ---: | ---: | ---: |
| Precision | 0.942301 | 0.955194 | +0.012893 |
| Recall | 0.903809 | 0.923329 | +0.019521 |
| mAP@0.5 | 0.953653 | 0.968137 | +0.014484 |
| mAP@0.5:0.95 | 0.607514 | 0.627392 | +0.019878 |
| MAE conteo (cabezas/imagen) | 0.357043 | 0.314505 | -0.042538 |

El MAE se redujo aproximadamente 11,9%. En este split WAID, la ampliación de
datos mejoró las métricas de detección y el conteo; esto no demuestra desempeño
en estancias paraguayas ni elimina la necesidad de validar en vuelos/escenas
independientes.

## Artefactos para reproducir

- Configuración: `configs/train_waid_cattle_1500.toml`.
- Modelo: `models/waid-cattle-yolo11n-1500.pt`.
- Metadatos: `models/waid-cattle-yolo11n-1500.metadata.json`.
- Evaluación: `reports/generated/waid-1500/evaluation.json`.
- Historial por época: `runs/waid/cattle-pilot-yolo11n-1500/results.csv`.

La siguiente evaluación de adaptación deberá usar datos paraguayos separados
por estancia/vuelo, reservando un test local independiente.
