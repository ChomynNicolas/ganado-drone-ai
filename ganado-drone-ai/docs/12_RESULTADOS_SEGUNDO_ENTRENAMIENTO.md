# Segundo entrenamiento YOLO11n sobre WAID

## Objetivo

Repetir el entrenamiento piloto del detector de ganado para comprobar que el
resultado es reproducible con el mismo checkpoint inicial, datos, semilla y
parámetros.

## Configuración

- Modelo inicial: `models/yolo11n.pt` (YOLO11n preentrenado en COCO).
- Datos: `data/processed/waid-cattle-clean-v1`.
- Entrenamiento: 50 épocas, tamaño de imagen 640, batch 8, paciencia 10.
- Semilla: 42; workers: 0; dispositivo: NVIDIA GeForce RTX 4060 Laptop GPU.
- Entorno: Python 3.11.9, PyTorch 2.9.1+cu126, Ultralytics 8.4.198.
- Mejor checkpoint: época 48.

## Dataset local

| Split | Imágenes | Cajas de ganado |
| --- | ---: | ---: |
| Train | 625 | 6.608 |
| Validación | 2.887 | 12.798 |
| Test | 1.434 | 6.234 |

El conjunto WAID procesado permanece local y no forma parte de este commit.

## Métricas

### Mejor checkpoint en validación

| Métrica | Resultado |
| --- | ---: |
| Precision | 0.933770 |
| Recall | 0.895760 |
| mAP@0.5 | 0.947240 |
| mAP@0.5:0.95 | 0.601140 |

### Evaluación final en test

| Métrica | Resultado |
| --- | ---: |
| Precision | 0.942301 |
| Recall | 0.903809 |
| mAP@0.5 | 0.953653 |
| mAP@0.5:0.95 | 0.607514 |
| MAE de conteo por imagen | 0.357043 cabezas |
| Umbral de confianza seleccionado en validación | 0.40 |

El MAE mide el error medio de conteo por fotografía; no estima individuos
únicos entre imágenes superpuestas.

## Comparación con la primera corrida

Las 499 entradas tensoriales de los checkpoints de ambas corridas son iguales
valor por valor. Coinciden también las métricas de validación y test. Esto
confirma que el entrenamiento es reproducible con esta receta, pero no
representa una mejora del modelo ni una evaluación independiente: se utilizaron
los mismos datos y particiones.

Los archivos serializados tienen SHA-256 distintos porque el hash cubre el
archivo checkpoint completo, no solamente los tensores del modelo. La
comparación tensorial confirmó que los pesos aprendidos son iguales.

## Artefactos incluidos

- Configuración reproducible: `configs/train_waid_cattle_v2.toml`.
- Pesos del segundo checkpoint: `models/waid-cattle-yolo11n-v2.pt`.
- Metadatos y hash del checkpoint: `models/waid-cattle-yolo11n-v2.metadata.json`.
- Métricas formales de test: `reports/generated/evaluation-v2.json`.
- Historial por época: `runs/waid/cattle-pilot-yolo11n-v2/results.csv`.

El peso del modelo se distribuye en este commit. Las imágenes WAID y sus
etiquetas no se redistribuyen desde el repositorio.
