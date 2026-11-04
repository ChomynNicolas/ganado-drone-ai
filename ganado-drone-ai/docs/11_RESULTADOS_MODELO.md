# Resultados del modelo WAID

## Protocolo

Se entrenó YOLO11n con una clase (`cattle`) a partir del piloto WAID preparado en `data/processed/waid-cattle-clean-v1`. Las particiones se agruparon por fotografía fuente. Entrenamiento: 625 imágenes, 6 608 cajas; validación: 2 887 imágenes, 12 798 cajas; test: 1 434 imágenes, 6 234 cajas. Total: 4 946 imágenes y 25 640 cajas. Se descartaron 51 cajas degeneradas de ovinos y el control de etiquetas no encontró errores ni grupos de fuente cruzados entre splits.

La revisión WAID utilizada corresponde al commit `f6de7fb1fdd78cd794b31526dde0eb37fd321e38` del repositorio fuente.

Configuración: 50 épocas, imagen 640, batch 8, patience 10, semilla 42, workers 0. Entorno: Python 3.11.9, Ultralytics 8.4.198, PyTorch 2.9.1+cu126 y NVIDIA GeForce RTX 4060 Laptop GPU. El umbral de conteo 0.40 se eligió por MAE en `valid`; el conjunto `test` se reservó para la evaluación final.

## Evaluación en test

| Métrica | Resultado |
| --- | ---: |
| Precision | 0.9423 |
| Recall | 0.9038 |
| mAP@0.5 | 0.9537 |
| mAP@0.5:0.95 | 0.6075 |
| MAE de conteo por imagen | 0.3570 |

Los resultados corresponden a 1 434 imágenes de test. La latencia observada en GPU fue aproximadamente 3.1 ms de inferencia, 1.2 ms de preprocesamiento y 0.7 ms de postprocesamiento por imagen. Esta medida no incluye transporte HTTP ni renderizado del dashboard. El MAE es por fotografía: no cuenta individuos únicos entre imágenes superpuestas.

## Artefactos y reproducción

- Checkpoint local: `models/waid-cattle-yolo11n.pt`.
- SHA-256: `65fbdb3e3fddd9a2db715529656d45152f945b5fb1347037a66dcd5ee771b1a4`.
- Parámetros: `configs/train_waid_cattle.toml`.
- Manifiesto y pesos se conservan localmente; WAID no se redistribuye desde este repositorio.
- Evaluación: `reports/generated/evaluation.json`.
- Directorio de entrenamiento: `runs/waid/cattle-pilot-yolo11n/`.

Repetir desde el entorno configurado:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_yolo --config configs/train_waid_cattle.toml
.\.venv\Scripts\python.exe -m scripts.evaluate_yolo --weights models/waid-cattle-yolo11n.pt --dataset data/processed/waid-cattle-clean-v1 --output reports/generated/evaluation.json --imgsz 640 --device auto
```

## Límites

Es un piloto sobre WAID, no una evaluación en vuelos paraguayos. La precisión medida no demuestra generalización a otras cámaras, alturas, regiones o condiciones. Debe completarse la revisión visual a resolución completa y confirmarse la licencia antes de redistribuir datos o pesos. El baseline COCO de humo no es una comparación cuantitativa. El Compose CPU fue iniciado y la API/dashboard respondieron; el overlay GPU en Docker aún requiere validación en el equipo objetivo.
