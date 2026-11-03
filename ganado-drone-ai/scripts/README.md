# Herramientas disponibles

- `scripts/check_environment.py`: valida versión de Python y herramientas de desarrollo.
- `scripts/validate_yolo_labels.py`: valida etiquetas ya convertidas a una clase (`cattle=0`).
- `scripts/audit_waid.py`: inspecciona la estructura original WAID, clases (`cattle=1`), etiquetas por split y duplicados exactos opcionales.

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
```

Todos estos scripts pueden usarse sin PyTorch. No entrenan modelos y no modifican datos brutos.
