# Cambio 06 — Selección y adquisición de datos WAID

## Decisión de fuente

Se propone WAID (Mou et al., 2023) como **dataset público de referencia para auditar** y eventual entrenamiento. La publicación oficial informa imágenes aéreas de animales y el repositorio fuente expone etiquetas YOLO. El archivo `classes.txt` oficial establece **`cattle=1`**, no `0`.

- Publicación: https://doi.org/10.3390/app131810397
- Repositorio oficial: https://github.com/xiaohuicui/WAID
- Catálogo independiente: https://github.com/agentmorris/drone-wildlife-datasets

La licencia del **repositorio oficial** no pudo confirmarse. Una copia de terceros en Roboflow con licencia CC BY 4.0 no demuestra que el repositorio fuente tenga esa licencia. No subir fotos ni compartir dataset derivado hasta aclarar los términos de uso.

## 1. Obtener la fuente sin importarla al Git del proyecto

En PowerShell, desde la raíz de `ganado-drone-ai`, después de publicar el Cambio 06:

```powershell
# Puede superar 1 GB; verificar conexión y espacio disponible.
git clone --depth 1 --filter=blob:none https://github.com/xiaohuicui/WAID.git data/raw/waid-repo

# Guardar este identificador en la ficha al terminar correctamente.
git -C data/raw/waid-repo rev-parse HEAD

# Debe mostrar 'true' para confirmar que está ignorado por el Git del proyecto.
git check-ignore data/raw/waid-repo/WAID/classes.txt
```

Si el servidor no admite el filtro parcial, Git puede descargar la historia superficial completa. No confundir fallos de red o descarga incompleta con dataset vacío.

Los datos locales quedarán en:

```text
data/raw/waid-repo/WAID/classes.txt
data/raw/waid-repo/WAID/images/train/
data/raw/waid-repo/WAID/images/valid/
data/raw/waid-repo/WAID/images/test/
data/raw/waid-repo/WAID/labels/train/
data/raw/waid-repo/WAID/labels/valid/
data/raw/waid-repo/WAID/labels/test/
```

**Alternativa:** descargar el ZIP desde la página oficial de GitHub y extraerlo bajo `data/raw/waid-repo/`; en ese caso registrar la URL y fecha del ZIP, y no utilizar `rev-parse` si no hay `.git`.

## 2. Ejecutar auditoría solo lectura

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
```

Resultado: conteo por split, imágenes y cajas con ganado, otras clases, etiquetas huérfanas, anotaciones malformadas, duplicación exacta entre splits y advertencias de integridad. El JSON se escribe en `reports/generated/`, ignorado por Git.

**Criterio de aceptación:** sin errores críticos en el informe; revisar advertencias, registrar versión de origen y mostrar al menos algunas muestras de imágenes con sus cajas manualmente. El hash solo comprueba duplicados exactos y **no** garantiza ausencia de leakage entre vuelos similares.

## 3. Evidencias a documentar después de verificar la descarga

- SHA del commit de WAID, fecha real de adquisición y ubicación de la fuente.
- Cantidad real de imágenes y cajas `cattle` por cada partición del dataset.
- Errores encontrados, decisión de corrección y evidencia de inspección visual.
- Riesgo de fuga de información por imágenes relacionadas y procedencia regional.
- Confirmación de licencia o restricción documental explícita.

El auditor utiliza etiquetas **originales de seis clases**. El validador del Cambio 05 continuará siendo para **datos procesados de una clase**. La conversión a `cattle=0`, el muestreo de entrenamiento y `dataset.yaml` se implementarán en el Cambio 07.
