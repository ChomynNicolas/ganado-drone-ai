# ADR-001 — Stack para detección y conteo aéreo de ganado

- **Fecha:** 2026-11-03
- **Estado:** Aceptado para la primera implementación; sujeto a benchmarks y disponibilidad de datos
- **Responsable de registro:** equipo del Proyecto 2

## Contexto

La consigna exige detección de objetos sobre fotografías aéreas, métricas formales, API REST, UI, Git y demo. El equipo dispone de hardware local limitado (GTX 1650, 4 GB VRAM) y necesita validar rápidamente un pipeline end-to-end sin priorizar componentes que no aportan al objetivo.

## Decisión

Adoptar **Python 3.11 + PyTorch/Ultralytics YOLO11n + FastAPI + Streamlit + pytest**, con Docker Compose como objetivo de despliegue. Entrenar una sola clase `cattle` usando transfer learning y evaluar cajas y conteo. Empezar con el checkpoint `yolo11n.pt` **como baseline**, no como modelo final ajustado a fotografías aéreas.

La UI consume la API HTTP y no ejecuta inferencia directamente. No usar una base de datos. Trabajar incrementalmente en Git, con un commit por unidad lógica; por ahora sin requerir PR ni reviewer.

## Alternativas consideradas

| Alternativa | Motivo para no seleccionarla en el MVP |
|---|---|
| Faster R-CNN | pipeline de ajuste e integración comparativamente más costoso para el tiempo/equipo disponible |
| YOLO medianos/grandes | mayor demanda de VRAM y latencia; pasar a ellos solo con métricas que justifiquen el cambio |
| Next.js/React como UI | agrega servidor Node/build y trabajo UI sin ventaja indispensable para la demo de inferencia |
| Gradio como UI | viable, pero Streamlit ofrece control suficiente sobre resumen/tablas y flujo de lotes; elegir una sola UI |
| PostgreSQL + ORM | el MVP es stateless y no requiere almacenamiento de usuarios/historial |
| Video + tracking | cambia el problema; la consigna solicita fotografías aéreas y conteo sobre cajas |

## Consecuencias y trade-offs

**A favor:** un único lenguaje para ML, backend e interfaz; menos integración; documentación OpenAPI; demo offline posible; modelo compacto y fácil de comparar con un fine-tuning.

**Limitaciones:** el detector generalista puede fallar con animales de tamaño reducido; tiling puede aumentar complejidad y duplicados; Streamlit no es el frontend adecuado para un SaaS complejo; la GPU de 4 GB limita batch/resolución; Docker en CPU puede ser lento.

**Legalidad:** Ultralytics distribuye bajo condiciones AGPL-3.0 / opciones comerciales. Verificar licencias de código, pesos y dataset antes de hacer público el repositorio o redistribuir artefactos. No se declara una licencia propia hasta resolverlo.

## Cómo se verificará la decisión

1. Confirmar el entorno Python y la disponibilidad real de GPU (`torch.cuda.is_available()`).
2. Verificar dataset y etiquetas; WAID es candidato, todavía no selección confirmada.
3. Medir inferencia baseline en imágenes reales y documentar fallos; registrar hardware.
4. Ajustar el modelo y comparar mAP@0.5, mAP@0.5:0.95, precisión, recall y MAE de conteo usando particiones sin leakage.
5. Validar API + dashboard + despliegue local con un lote real y un checkpoint entrenado.

## Fuentes

- Catálogo de la materia: `Proyecto_ia_para_el_2026.docx`, Proyecto 2 y evaluación.
- <https://docs.ultralytics.com/models/yolo11/>
- <https://docs.ultralytics.com/>
- <https://pytorch.org/get-started/locally/>
- <https://fastapi.tiangolo.com/tutorial/request-files/>
- <https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader>
