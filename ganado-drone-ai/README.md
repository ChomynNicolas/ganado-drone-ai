# Ganado Drone AI

**Proyecto 2 — Conteo y Monitoreo de Ganado Vacuno mediante Fotografías Aéreas de Drones**

Proyecto académico de Inteligencia Artificial (4.º año, Ingeniería Informática, Paraguay).

**Integrantes:** Nicolás Chomynn y Sebastián Paredes.

## Problema

Diseñar un sistema de visión por computadora que detecte y contabilice ganado bovino en fotografías aéreas tomadas por drones. El objetivo es apoyar tareas de inventario de animales en establecimientos ganaderos.

## Resultado esperado

Una aplicación que permita subir un lote de imágenes aéreas y visualizar las detecciones con cajas delimitadoras, cantidad por imagen y un total acumulado de detecciones del lote.

> Si las fotografías se superponen, una vaca puede aparecer en varias imágenes. La suma de detecciones no constituye un censo de animales únicos.

## Estado actual

**Stack definido y herramientas iniciales de datos implementadas.** Se documentó WAID y se agregaron un auditor, un conversor a `cattle=0` y una visualización de etiquetas. **WAID ya fue descargado y auditado localmente:** 14.366 imágenes, 51 anotaciones sheep degeneradas y 788 IDs base compartidos entre splits originales. El dataset piloto de una clase fue generado y pasó los controles automáticos. Hay código inicial para inferencia baseline YOLO11n sobre fotografías; el entrenamiento específico sobre WAID y la API/UI siguen pendientes.

## Desarrollo local

**Requisito:** Python 3.11 instalado.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check scripts tests
.\.venv\Scripts\python.exe -m ruff format --check scripts tests
```

## Dataset candidato: WAID

Fuente oficial: https://github.com/xiaohuicui/WAID

La fuente original dispone de imágenes/etiquetas en `train`, `valid` y `test` y `classes.txt` declara `cattle=1`. **No usar el validador de una clase sobre etiquetas originales.**

Descarga, auditoría, licencia y reproducibilidad: [`docs/06_FUENTE_WAID.md`](docs/06_FUENTE_WAID.md).

```powershell
git clone --depth 1 --filter=blob:none https://github.com/xiaohuicui/WAID.git data/raw/waid-repo
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
```

Los datasets descargados quedan en `data/raw/`, ignorado por Git. Los resultados locales de auditoría y preparación se registran con `scripts/record_dataset_results.py` en `docs/09_RESULTADOS_DATASET.md`. No son métricas del modelo.

### Preparación para una clase (Cambios 07 y 08)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m scripts.prepare_cattle_dataset --source data/raw/waid-repo/WAID --output data/processed/waid-cattle-clean-v1 --max-train-positives 500 --seed 42 --allow-invalid-sheep-zero-boxes --resplit-by-source-image
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/waid-cattle-clean-v1/labels
.\.venv\Scripts\python.exe -m scripts.preview_cattle --dataset data/processed/waid-cattle-clean-v1 --split train --count 12 --output reports/generated/cattle-preview.jpg
```

La conversión reagrupa las variantes de una misma fotografía a un solo split y solo permite excluir con una opción explícita las cajas degeneradas conocidas de `sheep=0`; `valid/test` mantienen todos los ejemplos etiquetados, incluidos negativos. Una carpeta de salida existente **no se sobrescribe**. Detalles, comandos y advertencias: [`docs/07_PREPARACION_WAID_CATTLE.md`](docs/07_PREPARACION_WAID_CATTLE.md). Incidencia identificada: [`docs/08_INCIDENCIA_CALIDAD_WAID.md`](docs/08_INCIDENCIA_CALIDAD_WAID.md).

### Registro reproducible de la preparación (Cambio 09)

```powershell
.\.venv\Scripts\python.exe -m scripts.record_dataset_results --manifest data/processed/waid-cattle-clean-v1/manifest.json --groups reports/generated/split_group_audit.json --output docs/09_RESULTADOS_DATASET.md
```

Solo se publica el resumen Markdown; los JSON e imágenes originales siguen fuera de Git.

### Baseline COCO (Cambio 10)

El detector preentrenado de COCO reconoce la clase `cow` (ID 19), distinta de la clase `cattle=0` del dataset procesado. No constituye un modelo entrenado sobre fotografías aéreas.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m scripts.download_baseline
.\.venv\Scripts\python.exe -m scripts.infer_baseline --source data/processed/waid-cattle-clean-v1/images/train --limit 5 --device auto
```

La instalación de PyTorch CUDA se explica en [`docs/10_BASELINE_YOLO.md`](docs/10_BASELINE_YOLO.md); se recomienda hacerlo antes de instalar el resto de requisitos ML. Los pesos, anotaciones, fotografías y reportes de inferencia no se publican en Git.

## Estructura y documentación

- `api/app/`: backend aún no implementado.
- `ui/`: dashboard aún no implementado.
- `data/DATASET_CARD.md`: fuente, etiquetas, restricciones y evidencia pendiente.
- `configs/`: futura configuración de entrenamiento.
- `models/`: checkpoints locales ignorados por Git.
- `reports/`: informe y métricas; `reports/generated/` está ignorado.
- `scripts/`: herramientas de control del entorno y datos.
- `tests/`: pruebas sintéticas de scripts; **no** validación real del modelo.
- [`docs/01_ALCANCE.md`](docs/01_ALCANCE.md): consigna y alcance.
- [`docs/02_BACKLOG_INICIAL.md`](docs/02_BACKLOG_INICIAL.md): actividades y criterios de aceptación.
- [`docs/03_DECISIONES_TECNICAS.md`](docs/03_DECISIONES_TECNICAS.md): selección de tecnologías.
- [`docs/05_ARQUITECTURA_TECNICA.md`](docs/05_ARQUITECTURA_TECNICA.md): componentes previstos.
- [`docs/06_FUENTE_WAID.md`](docs/06_FUENTE_WAID.md): fuente de imágenes y adquisición.
- [`docs/07_PREPARACION_WAID_CATTLE.md`](docs/07_PREPARACION_WAID_CATTLE.md): conversión reproducible e inspección visual.
- [`docs/08_INCIDENCIA_CALIDAD_WAID.md`](docs/08_INCIDENCIA_CALIDAD_WAID.md): diagnóstico y corrección de anotaciones y splits.
- [`docs/09_RESULTADOS_DATASET.md`](docs/09_RESULTADOS_DATASET.md): evidencia local generada automáticamente y versionada.
- [`docs/10_BASELINE_YOLO.md`](docs/10_BASELINE_YOLO.md): instalación, pesos y prueba de inferencia COCO.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): historial Git y colaboración.

## Trabajo en Git

Nicolás registra cambios pequeños comprobados en `main` y Sebastián utiliza su cuenta e identidad Git cuando contribuya. Las imágenes, las etiquetas y los pesos entrenados se mantienen fuera del repositorio. Los Issues pueden documentar cambios sin PR obligatorios en esta etapa.

## Entregables pendientes

Modelo con fine-tuning reproducible, API REST, interfaz para lotes, Docker Compose, informe IEEE de exactamente tres páginas y demostración en vivo. Nada de esto debe declararse como terminado hasta que haya evidencia de ejecución.
