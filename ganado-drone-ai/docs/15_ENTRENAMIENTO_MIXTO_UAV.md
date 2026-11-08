# Entrenamiento mixto WAID + imágenes UAV

## Objetivo y estado

Este experimento ajusta el checkpoint WAID de 5.000 imágenes para mejorar su respuesta ante imágenes aéreas de otro origen. Se conserva el checkpoint base como referencia. El entrenamiento se ejecuta localmente; las imágenes, etiquetas, pesos derivados y salidas permanecen fuera de Git.

## Datos y particiones

- WAID aporta las 5.000 imágenes de entrenamiento de `waid-cattle-5000-v1`.
- Dataset 1 UAV aporta únicamente las 316 imágenes con cajas `cattle` verificadas: 1.887 cajas en total. Se excluyen 340 imágenes sin cajas, porque el paquete no las marca como negativas.
- La lista de entrenamiento contiene 5.000 exposiciones WAID y 1.250 UAV por época (80/20). Las fotos UAV se repiten de forma determinista con semilla 42 para alcanzar la proporción.
- Los identificadores de animales aparecen en más de un vuelo de Dataset 1. Por ese motivo no se separa ese conjunto aleatoriamente: la validación y selección de umbral se hacen con WAID valid.
- Se reservan WAID test, Dataset 2 (14 imágenes, 62 cajas) y `cow_images` (655 imágenes, anotaciones puntuales) para la evaluación final. Ninguno se usa para entrenar o calibrar.

## Conversión y entrenamiento

Las cajas Supervisely se convierten a una clase YOLO (`cattle=0`) en `scripts.prepare_uav_mixed_dataset`. La preparación confirma el emparejamiento imagen-etiqueta, dimensiones, licencia declarada y coordenadas. Los archivos originales no se modifican.

Configuración: checkpoint inicial WAID de 5.000 imágenes; máximo 50 épocas; resolución 640; lote 8; paciencia 10; semilla 42; dispositivo automático. La selección del mejor checkpoint sigue la validación WAID de Ultralytics.

## Evaluación prevista

Se comparan el checkpoint WAID-5000 y el checkpoint mixto con IoU 0,50, mAP@0.5, mAP@0.5:0.95 y MAE de conteo. El umbral operativo se escoge en WAID valid como el de mayor recall con precisión de cajas de al menos 90%; también se registra el comportamiento a confianza 0,40.

Dataset 2 se evalúa con cajas y mAP. `cow_images` se evalúa como conjunto externo de Nueva Zelanda mediante coincidencia de puntos, precisión, recall, F1 y MAE de conteo; sus etiquetas no permiten calcular mAP de cajas. Los resultados de estas fuentes no representan desempeño en Paraguay.

## Resultados de la corrida completada

La corrida completó las 50 épocas en una NVIDIA RTX 4060, con entrada 640, batch 8 y semilla 42. El mejor checkpoint de validación corresponde a la época 49 (precisión 0,9657; recall 0,9401; mAP@0.5 0,9736; mAP@0.5:0.95 0,6449). El checkpoint final se guardó localmente como `models/waid-cattle-yolo11n-uav-mixed.pt` y no se activó en Docker.

El umbral escogido solo con validación WAID fue 0,25 para ambos checkpoints, con precisión de cajas mínima de 90% e IoU 0,50. El test WAID se usó después de congelar ambos modelos. Dataset 2 y `cow_images` son evaluaciones externas; ninguna imagen de esas fuentes participó en entrenamiento o selección de umbral.

| Evaluación | WAID-5000 | WAID + UAV | Lectura |
|---|---:|---:|---|
| WAID test: mAP@0.5 | 0,9760 | 0,9769 | +0,0009 |
| WAID test: mAP@0.5:0.95 | 0,6448 | 0,6500 | +0,0052 |
| WAID test: MAE de conteo, umbral 0,25 | 0,3898 | 0,3159 | Baja 0,0739 animales/foto |
| WAID test: falsos positivos por foto negativa, umbral 0,25 | 0,0031 | 0,0062 | Sube 0,0031; revisar negativos |
| WAID test: MAE de conteo, confianza 0,40 | 0,2057 | 0,1834 | Baja 0,0223 animales/foto |
| Dataset 2: mAP@0.5 | 0,0246 | 0,4461 | Mejora externa grande |
| Dataset 2: mAP@0.5:0.95 | 0,0060 | 0,1482 | Mejora externa grande |
| Dataset 2: MAE de conteo, confianza 0,40 | 4,3571 | 2,8571 | Baja 1,5 animales/foto |
| `cow_images`: recall por puntos, confianza 0,40 | 0,0561 | 0,2231 | Sube 0,1670 |
| `cow_images`: F1 por puntos, confianza 0,40 | 0,1060 | 0,3622 | Sube 0,2562 |
| `cow_images`: MAE de conteo, confianza 0,40 | 42,8656 | 35,0168 | Baja 7,8489 animales/foto |

En `cow_images`, el recall aumentó junto con más cajas falsas: FP pasó de 62 a 260 y la precisión pasó de 0,9642 a 0,9624. El modelo mixto reduce el subconteo, pero aún omite muchos animales (sesgo medio −34,95 detecciones/foto). En Dataset 2, a confianza 0,40, recall de cajas subió de 0 a 0,2903 y se detectaron 18 de 62 cajas; el resultado sigue lejos de una detección completa. Esta muestra solo contiene 14 imágenes.

La mejora externa es prometedora, pero no prueba rendimiento en Paraguay y no elimina los falsos positivos. WAID test mantiene rendimiento alto; su tasa de falsos positivos por foto negativa aumenta al umbral calibrado 0,25, aunque a 0,40 ambos checkpoints dan 0,0010. Se recomienda usar 0,40 como umbral conservador para la siguiente demo y mantener el modelo mixto en evaluación local hasta revisar más imágenes negativas.

Los reportes completos, los errores por imagen y las visualizaciones están en `reports/generated/waid-uav-mixed/` y permanecen ignorados por Git por la licencia y la procedencia de los datos. El historial de Ultralytics conserva filas duplicadas en algunas épocas reanudadas; las métricas de validación por época coinciden y el checkpoint elegido se verificó contra su evaluación WAID. La columna de tiempo de `results.csv` no debe usarse para afirmar duración total.

## Licencia

El paquete de Dataset 1 declara CC BY-NC-ND 4.0. Este experimento es local y no se publicarán imágenes, anotaciones ni pesos derivados sin revisar previamente los permisos aplicables. Citar a Shao et al., *Cattle detection and counting in UAV images based on convolutional neural networks*, DOI: 10.1080/01431161.2019.1624858.
