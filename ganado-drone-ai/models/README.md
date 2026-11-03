# Artefactos del detector

Aquí se colocarán los checkpoints entrenados después de reproducir los experimentos.
Los archivos `*.pt`, `*.pth`, `*.onnx` y `*.engine` están excluidos de Git.

No existe todavía un modelo entrenado o verificado para este proyecto.
Cuando exista, registrar origen, licencia, hash SHA-256, dataset, parámetros,
métricas y comando para reproducir su obtención. La demo offline necesitará
una copia local explícita del checkpoint; documentar ese paso sin incluirlo
automáticamente en Git.

## Primer checkpoint descargable

`python -m scripts.download_baseline` obtiene `models/yolo11n.pt` (YOLO11n preentrenado COCO). El modelo final fine-tuned sobre WAID todavía no existe. Guardar SHA-256 del checkpoint y las versiones de PyTorch/Ultralytics usadas durante la ejecución.
