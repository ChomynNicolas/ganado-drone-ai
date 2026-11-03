# Registro de decisiones técnicas

No confundir una **propuesta** con una **decisión cerrada**.

| Decisión | Estado | Criterio principal |
|---|---|---|
| Detector preentrenado (familia YOLO pequeña u otra) | Pendiente | objetos pequeños, facilidad de entrenamiento e inferencia |
| Fuente del dataset | Pendiente | anotaciones, licencia, variedad de altura y perspectiva |
| Estrategia de split | Pendiente | evitar leakage de imágenes de un mismo vuelo/escena |
| Backend REST | Pendiente (FastAPI propuesto) | integración simple con Python y modelo |
| Interfaz | Pendiente (Streamlit propuesto) | rapidez y estabilidad de carga de lotes |
| Contenedores | Pendiente | CPU para demo reproducible, peso del artefacto |

## Plantilla ADR

```md
# ADR-001: [decisión]
Fecha: [YYYY-MM-DD]
Estado: Propuesto | Aceptado | Reemplazado
Contexto: ...
Alternativas: ...
Decisión: ...
Consecuencias / trade-offs: ...
Evidencia: [issue, PR, benchmark, fuente]
```

Guardar los ADR aceptados en `docs/adr/` cuando existan decisiones reales.
