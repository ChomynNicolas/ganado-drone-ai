# Registro de decisiones técnicas

**Estado:** arquitectura propuesta para el MVP, sin implementación ni validación de rendimiento todavía.

Este documento separa decisiones **adoptadas** de hipótesis **pendientes de verificar**. La justificación central está en [`adr/ADR-001-stack.md`](adr/ADR-001-stack.md); el contrato y el flujo técnico están en [`05_ARQUITECTURA_TECNICA.md`](05_ARQUITECTURA_TECNICA.md).

## Decisiones adoptadas para implementar

| Área | Decisión | Motivo | Estado |
|---|---|---|---|
| Lenguaje | Python 3.11 | ecosistema de visión artificial y despliegue sencillo | Adoptada |
| Tarea ML | Detección de objetos, una clase (`cattle`) | cajas delimitadoras + conteo por imagen | Adoptada |
| Detector inicial | Ultralytics YOLO11n (`yolo11n.pt`) | baseline preentrenado, tamaño pequeño y fine-tuning | Adoptada para primer experimento |
| Librería ML | PyTorch / Ultralytics | entrenamiento, validación e inferencia integrados | Adoptada |
| API | FastAPI + Pydantic | contrato HTTP, validación y OpenAPI automáticos | Adoptada |
| Interfaz | Streamlit | carga de múltiples fotos y demo local rápida | Adoptada |
| Pruebas | pytest | regresión en funciones de datos y endpoints | Adoptada |
| Contenedores | Docker Compose, API y UI separados | despliegue reproducible y demo sin servicios externos | Adoptada como objetivo |
| Persistencia | Sin base de datos en MVP | no se requieren cuentas ni almacenamiento histórico | Adoptada |
| Control de versiones | Git/GitHub, commits incrementales en `main` | trazabilidad sin revisiones por ahora | Adoptada temporalmente |

**Nota sobre el detector:** el nombre del modelo no implica que ya esté entrenado para reconocer vacas vistas desde drones. Se evaluará el checkpoint preentrenado como baseline y posteriormente un checkpoint ajustado al dataset elegido.

## Decisiones pendientes / gates de validación

| Tema | Estado | Condición para cerrar |
|---|---|---|
| Dataset | WAID es candidato, **no elegido todavía** | verificar fuentes, licencia, clases, calidad y distribución de escenas |
| Índice de clase origen | Pendiente | inspeccionar el YAML/etiquetas reales; no asumir un `class_id` |
| Split final | Pendiente | evitar imágenes del mismo vuelo/escena en grupos distintos, cuando haya metadata |
| Tamaño de imagen y batch | Pendiente | medir memoria y recall con GTX 1650 (4 GB); alternativa Colab |
| Tiling / slicing | Condicional | agregar solo si las pruebas muestran pérdida de animales pequeños |
| Versiones exactas de paquetes | Pendiente | fijarlas al crear el entorno y registrar `pip freeze` |
| Licencias | Pendiente | comprobar permiso del dataset y condiciones de Ultralytics antes de publicar artefactos |
| Umbral de confianza | Pendiente | seleccionar con validación, nunca con test |
| Hardware de la demo | Pendiente | medir latencia en CPU y confirmar ejecución offline |

## Protocolo de evaluación aprobado

- Baseline: detector preentrenado con la misma tarea de salida; evaluar de forma comparable, sin atribuirle un entrenamiento propio.
- Fine-tuning: ajustar el modelo con las imágenes **train** y usar **validation** para selección de parámetros.
- Test: reservado para evaluación final, sin usarlo iterativamente para elegir el mejor checkpoint o umbral.
- Métricas: mAP@0.5, mAP@0.5:0.95, precision, recall, MAE de conteo por fotografía y latencia por imagen con hardware identificado.
- Reproducibilidad: registrar dataset/versión, seed, parámetros, versiones de bibliotecas, hardware y checkpoints.
- Interpretación: `sum(count_per_image)` informa **detecciones acumuladas**, no cantidad de bovinos únicos si hay solapamiento.

## Fuentes técnicas

- Documentación de modelos: <https://docs.ultralytics.com/models/yolo11/>
- API de FastAPI: <https://fastapi.tiangolo.com/tutorial/request-files/>
- Archivos múltiples Streamlit: <https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader>
- Instalación PyTorch: <https://pytorch.org/get-started/locally/>
- Candidato WAID: <https://github.com/xiaohuicui/WAID>
- Licencias Ultralytics: <https://docs.ultralytics.com/>
