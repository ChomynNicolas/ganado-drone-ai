# Incidencia IA-DATA-01 — Calidad y particiones WAID

**Fecha del diagnóstico:** 2026-11-03. **Estado:** corrección implementada sobre el conversor; pendiente de ejecutar con WAID real y verificar los resultados.

## Evidencia recibida

Auditoría `python -m scripts.audit_waid --hash-check` del dataset descargado:

| Split original | Imágenes | Imágenes cattle | Cajas cattle válidas | TXT ausentes |
|---|---:|---:|---:|---:|
| train | 10.056 | 3.267 | 44.245 | 0 |
| valid | 2.873 | 932 | 12.605 | 0 |
| test | 1.437 | 466 | 6.240 | 0 |

El auditor registró **51 errores de anotación**. La inspección posterior mostró que los 51 corresponden a **clase `sheep=0`** con ancho o alto no positivo, por ejemplo:

```text
0 0.24583333333333332 0.27416666666666667 0 0.0008333333333333334
```

El análisis de nombres originales (antes de `.rf.`) identificó **788 identificadores presentes en más de un split**. Son candidatos fuertes a variantes aumentadas de una misma foto. Esto invalida el uso ingenuo del split para medir generalización independiente.

## Política de corrección (sin modificar los originales)

1. El auditor original sigue siendo estricto: sus 51 errores **no se ocultan**.
2. Solo con `--allow-invalid-sheep-zero-boxes`, el conversor tolera y descarta **líneas de `sheep=0` cuyo ancho o alto es exactamente cero**, y solo cuando las demás coordenadas están normalizadas y son finitas. Otras anotaciones inválidas, especialmente `cattle=1`, provocan error.
3. Solo con `--resplit-by-source-image`, el conversor redistribuye **grupos de variantes** identificadas por el nombre anterior a `.rf.`. Se estratifican los grupos según presencia de cattle; distribución objetivo por número de grupos 70/20/10, con `seed=42`.
4. Si se intenta conservar el split original con grupos superpuestos, el conversor aborta. La configuración de exclusión y la lista de errores aceptados quedan en `manifest.json`.
5. El dataset original en `data/raw` se conserva intacto. Los datos derivados se generan en una carpeta nueva ignorada por Git.

## Comandos de reproducción

```powershell
$revision = git -C data/raw/waid-repo rev-parse HEAD

.\.venv\Scripts\python.exe -m scripts.prepare_cattle_dataset `
  --source data/raw/waid-repo/WAID `
  --output data/processed/waid-cattle-clean-v1 `
  --allow-invalid-sheep-zero-boxes `
  --resplit-by-source-image `
  --max-train-positives 500 `
  --train-negative-ratio 0.25 `
  --seed 42 `
  --source-revision $revision
```

Luego verificar `manifest.json`, las cajas, los grupos de identificadores y crear mosaicos de inspección para train, valid y test.

```powershell
.\.venv\Scripts\python.exe -m scripts.check_split_groups `
  --dataset data/processed/waid-cattle-clean-v1 `
  --output reports/generated/split_group_audit.json

.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py `
  --labels data/processed/waid-cattle-clean-v1/labels `
  --num-classes 1
```

El checker debe imprimir **0 grupos en varios splits**. Su agrupación por nombre no identifica fotogramas vecinos ni garantiza independencia por vuelo. La distribución de grupos evita **el leakage explícito por variantes con la misma fotografía base**, pero **no garantiza** independencia de fotografías diferentes del mismo vuelo, escena o potrero. Para una evaluación rigurosa, usar agrupación por vuelo cuando existan metadatos fiables, o documentar esta limitación.

## Criterios de aceptación

- Originales sin modificación y `manifest.json` con número exacto de cajas degeneradas excluidas.
- Todos los TXT procesados válidos para una única clase `cattle=0`.
- Ningún identificador base anterior a `.rf.` aparece en más de un split procesado.
- Sin anomalías nuevas en auditoría; fotos/etiquetas verificadas mediante inspección visual de los tres splits.
- Test reservado para la evaluación final, nunca para seleccionar hiperparámetros.

**Nota:** los resultados de procesamiento y conteos posteriores son pendientes hasta que se ejecute el pipeline con datos reales en la computadora del equipo.
