# Utilidades ejecutables

Utilidad actual: `scripts/check_environment.py`, que verifica Python y las
versiones de las herramientas de desarrollo.

Utilidad adicional: `scripts/validate_yolo_labels.py`, que inspecciona archivos
YOLO `.txt` después de convertir etiquetas originales a `cattle=0`.

```powershell
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/labels/train
```

El resultado no valida imágenes, licencias ni splits: esas comprobaciones quedan
pendientes. Scripts de conversión, entrenamiento y evaluación se incorporarán
cuando existan datos reales y código que pueda ejecutarse.
