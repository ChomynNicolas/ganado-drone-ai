# Backlog inicial — propuesta para GitHub Issues

**Importante:** las tareas enumeradas son sugerencias. No están completadas y no deben registrarse como logros hasta disponer de código, resultados y evidencias.

| ID local | Issue sugerido | Responsable sugerido | Evidencia de cierre |
|---|---|---|---|
| IA-01 | Elegir el stack y registrar decisiones | Ambos | ADR con criterios, alternativas y elección |
| IA-02 | Inicializar entorno Python y pruebas | Nicolás | instalación desde cero y prueba mínima verde |
| IA-03 | Seleccionar dataset y documentar licencia | Sebastián | ficha de dataset + imágenes inspeccionadas |
| IA-04 | Preparar datos y validar etiquetas | Sebastián | script repetible + reporte de integridad |
| IA-05 | Implementar baseline del detector | Sebastián | imágenes inferidas + configuración registrada |
| IA-06 | Entrenar o ajustar modelo | Sebastián | checkpoint, comando y logs de entrenamiento |
| IA-07 | Evaluar detector y conteo | Sebastián | mAP, precision/recall y MAE de conteo por imagen |
| IA-08 | Implementar API de inferencia | Nicolás | contrato OpenAPI + pruebas de endpoint |
| IA-09 | Implementar interfaz y carga por lotes | Nicolás | flujo real de subida y boxes visibles |
| IA-10 | Integrar API, UI y pesos del modelo | Ambos | prueba end-to-end documentada |
| IA-11 | Contenerizar servicios | Nicolás | `docker compose up --build` reproducible |
| IA-12 | Preparar informe IEEE y defensa | Ambos | PDF 3 páginas + demo practicada |

## Cómo crear un Issue

- **Título:** `[IA-03] Seleccionar y documentar dataset de ganado aéreo`.
- **Descripción:** contexto, trabajo, criterios de aceptación, archivos esperados.
- **Asignación:** una persona responsable, otra revisora.
- **Labels:** `data`, `model`, `api`, `ui`, `docs`, `bug`, `chore` (crear según necesidad).
- **Estado:** `Open` → PR asociado → `Closed` cuando la evidencia cumple los criterios.

Un Issue de investigación puede cerrarse con documentos y hallazgos. Un Issue de implementación requiere código o artefactos probados.
