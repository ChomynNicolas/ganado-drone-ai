# Scripts de control del proyecto

- `check_environment.py`: valida versión de Python y herramientas de desarrollo.
- `validate_yolo_labels.py`: valida etiquetas YOLO, incluida la salida de una clase `cattle=0`.
- `audit_waid.py`: audita etiquetas **originales** de WAID (`cattle=1`), estructura de splits y hashes opcionales.
- `prepare_cattle_dataset.py`: audita WAID con hashes y prepara una copia derivada de **una clase**, sin mezclar splits.
- `preview_cattle.py`: crea un mosaico de imágenes con bounding boxes, usando Pillow. Guardar en `reports/generated/`.

Para ejecutar las herramientas desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
.\.venv\Scripts\python.exe -m scripts.prepare_cattle_dataset --source data/raw/waid-repo/WAID --output data/processed/waid-cattle-pilot --max-train-positives 500 --seed 42
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/waid-cattle-pilot/labels
.\.venv\Scripts\python.exe -m scripts.preview_cattle --dataset data/processed/waid-cattle-pilot --split train --count 12 --output reports/generated/cattle-preview.jpg
```

Guía de decisiones y advertencias: [`docs/07_PREPARACION_WAID_CATTLE.md`](../docs/07_PREPARACION_WAID_CATTLE.md).

No modifica el dataset de origen, ni entrena un detector ni publica imágenes. Los datos derivados, checkpoints, vistas previas y resultados generados están excluidos del repositorio mediante `.gitignore`.
