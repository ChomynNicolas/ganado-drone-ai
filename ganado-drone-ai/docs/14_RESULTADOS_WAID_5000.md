# Cuarto entrenamiento: WAID con 5.000 imágenes

## Objetivo

Ampliar el conjunto de entrenamiento y reforzar la calibración frente a imágenes negativas y falsas detecciones, manteniendo los parámetros y la división por fotografía fuente de los experimentos anteriores.

## Preparación del conjunto

Se usó WAID, convertido a una clase (`cattle`). El conjunto de entrenamiento quedó en 5.000 imágenes: 3.266 positivas y 1.734 negativas. Validación conserva 2.887 imágenes (933 positivas y 1.954 negativas) y test conserva 1.434 (466 positivas y 968 negativas).

Las 1.500 imágenes del entrenamiento previo están incluidas en el nuevo conjunto de entrenamiento. Validación y test conservaron los mismos nombres, imágenes y etiquetas que la versión anterior. La auditoría encontró cero grupos de fotografías fuente compartidos entre train, validación y test. Las 51 cajas sheep degeneradas admitidas por la auditoría de origen se excluyeron; las 9.321 etiquetas procesadas pasaron la validación sin errores.

## Configuración

- Modelo inicial: `yolo11n.pt` (YOLO11n preentrenado en COCO).
- Épocas: 50; mejor época por mAP@0.5:0.95 de validación: 49.
- Resolución: 640; lote: 8; paciencia: 10; semilla: 42.
- Entrenamiento determinista, GPU NVIDIA GeForce RTX 4060 Laptop.
- Umbral de confianza seleccionado exclusivamente con validación: 0,40.

## Comparación en test

| Corrida | Imágenes de entrenamiento | Umbral | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 | MAE de conteo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primera | 625 | 0,40 | 94,23% | 90,38% | 95,37% | 60,75% | 0,357 |
| Segunda (réplica) | 625 | 0,40 | 94,23% | 90,38% | 95,37% | 60,75% | 0,357 |
| Tercera | 1.500 | 0,35 | 95,52% | 92,33% | 96,81% | 62,74% | 0,315 |
| Cuarta | 5.000 | 0,40 | 96,31% | 94,10% | 97,60% | 64,48% | 0,206 |

La primera y la segunda corrida fueron una réplica y no son evidencias independientes. Frente a la tercera corrida, la cuarta aumentó la precisión en 0,79 puntos porcentuales, el recall en 1,76, el mAP@0.5 en 0,78 y el mAP@0.5:0.95 en 1,74. El MAE de conteo disminuyó 34,6%.

## Comportamiento ante imágenes positivas y negativas

Con el umbral 0,40, en validación se detectó al menos un bovino en 933 de 933 imágenes positivas (100%) y hubo detecciones en 3 de 1.954 imágenes negativas (0,15%). En test se detectó al menos un bovino en 465 de 466 imágenes positivas (99,79%); 1 de 968 imágenes negativas tuvo alguna detección (0,10%), con cuatro cajas falsas en total. El MAE de conteo fue 0,248 en validación y 0,206 en test.

El MAE cuenta cajas por imagen y no individuos únicos en fotos superpuestas. Las tasas de falsos positivos corresponden a estos conjuntos finitos y no garantizan el mismo resultado en producción.

## Archivos del experimento

- Configuración: `configs/train_waid_cattle_5000.toml`.
- Modelo: `models/waid-cattle-yolo11n-5000.pt`.
- Metadatos y huella del modelo: `models/waid-cattle-yolo11n-5000.metadata.json`.
- Historial de épocas: `runs/waid/cattle-pilot-yolo11n-5000/results.csv`.
- Métricas y calibración: `reports/generated/waid-5000/evaluation.json`.
- Informe comparativo en PDF: `output/pdf/comparacion_entrenamientos_waid_5000.pdf`.

No se incluyen imágenes, etiquetas ni manifiestos del dataset. La revisión de imágenes casi duplicadas por vuelo/escena sigue pendiente. WAID no representa todavía condiciones paraguayas, por lo que la generalización local debe medirse con datos separados de vuelos o estancias del Paraguay.
