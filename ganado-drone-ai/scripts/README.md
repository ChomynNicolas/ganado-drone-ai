# Scripts de control del proyecto

- `check_environment.py`: valida versión de Python y herramientas de desarrollo.
- `validate_yolo_labels.py`: valida etiquetas YOLO, incluida la salida de una clase `cattle=0`.
- `audit_waid.py`: audita etiquetas **originales** de WAID (`cattle=1`), estructura de splits y hashes opcionales.
- `prepare_cattle_dataset.py`: audita WAID con hashes y prepara una copia derivada de **una clase**, reagrupando variantes del mismo origen en splits independientes.
- `check_split_groups.py`: comprueba que variantes de una fotografía base no crucen train/valid/test.
- `preview_cattle.py`: crea un mosaico de imágenes con bounding boxes, usando Pillow. Guardar en `reports/generated/`.

Para ejecutar las herramientas desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
.\.venv\Scripts\python.exe -m scripts.prepare_cattle_dataset --source data/raw/waid-repo/WAID --output data/processed/waid-cattle-pilot --max-train-positives 500 --seed 42 --allow-invalid-sheep-zero-boxes --resplit-by-source-image
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/waid-cattle-pilot/labels
.\.venv\Scripts\python.exe -m scripts.preview_cattle --dataset data/processed/waid-cattle-pilot --split train --count 12 --output reports/generated/cattle-preview.jpg
```

Guía de decisiones y advertencias: [`docs/07_PREPARACION_WAID_CATTLE.md`](../docs/07_PREPARACION_WAID_CATTLE.md).

No modifica el dataset de origen, ni entrena un detector ni publica imágenes. Los datos derivados, checkpoints, vistas previas y resultados generados están excluidos del repositorio mediante `.gitignore`.

La primera auditoría real de WAID reveló **51 cajas `sheep=0` degeneradas** y **788 IDs de foto original repartidos en más de un split**. No modificar `data/raw`: consultar [`docs/08_INCIDENCIA_CALIDAD_WAID.md`](../docs/08_INCIDENCIA_CALIDAD_WAID.md). Las opciones de limpieza y de regrouping deben invocarse explícitamente y se registran en el manifiesto.
