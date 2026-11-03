# Datasheet del dataset — pendiente de adquisición

**Estado:** ningún conjunto de imágenes descargado ni verificado en este repositorio.

| Aspecto | Registro |
|---|---|
| Dataset seleccionado | Pendiente: WAID es solo candidato |
| URL y versión/fecha de descarga | Pendiente |
| Licencia de uso y redistribución | Pendiente de verificación |
| Autores y procedencia | Pendiente |
| Zona geográfica / dominio de captura | Pendiente |
| Clases de origen y mapeo a `cattle=0` | Pendiente |
| Número de imágenes y anotaciones | Pendiente |
| Resolución y variabilidad (altura/iluminación) | Pendiente |
| Train/validation/test y criterio de separación | Pendiente |
| Riesgos de leakage por vuelo, escena o duplicados | Pendiente |
| Limitaciones para inferir en Paraguay | Pendiente |

## Criterios de aceptación del dataset

1. Inspeccionar imágenes y etiquetas originales manualmente antes de convertirlas.
2. Documentar licencia, fuente exacta y autorización para usar/publicar ejemplos.
3. Mapear clases de origen de manera explícita; **no asumir que vaca es clase 0**.
4. Verificar validez geométrica de anotaciones y revisar muestras visualmente.
5. Separar train, validation y test sin mezclar imágenes correlacionadas de una misma escena/vuelo.
6. Documentar hashes/versión del dataset y semilla del particionado cuando aplique.

## Almacenamiento

Conservar imágenes y etiquetas descargadas en `data/raw/` y datos transformados
en `data/processed/`, ambos ignorados por Git. Versionar **documentación y scripts**.
No publicar imágenes de terceros ni checkpoints sin revisar licencias y permisos.

## Herramienta de inspección implementada (Cambio 05)

Tras confirmar el mapeo de clases y convertir las anotaciones al formato de una clase,
validar los `.txt` de la carpeta `labels/train` (y repetir para val/test):

```powershell
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/labels/train
```

La comprobación solo evalúa **integridad sintáctica y geométrica básica**; no
sustituye la inspección visual, la auditoría del split o las métricas del detector.
No hay dataset seleccionado en este momento.
