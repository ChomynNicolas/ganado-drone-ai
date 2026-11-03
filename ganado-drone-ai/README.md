# Ganado Drone AI

**Proyecto 2 — Conteo y Monitoreo de Ganado Vacuno mediante Fotografías Aéreas de Drones**

Proyecto académico de Inteligencia Artificial (4.º año, Ingeniería Informática, Paraguay).

**Integrantes:** Nicolás Chomynn y Sebastián Paredes.

## Problema

Diseñar un sistema de visión por computadora que detecte y contabilice ganado bovino en fotografías aéreas tomadas por drones. El objetivo es apoyar tareas de inventario de animales en establecimientos ganaderos.

## Resultado esperado

Una aplicación que permita subir un lote de imágenes aéreas y visualizar las detecciones con cajas delimitadoras, cantidad por imagen y un total acumulado de detecciones del lote.

> **Limitación a validar:** si las fotografías se superponen, una vaca puede aparecer en varias imágenes. La suma de detecciones no debe presentarse como cantidad de animales únicos sin deduplicación demostrada.

## Estado del proyecto

**Alcance y arquitectura documentados.** El stack inicial está definido para construir el MVP, pero aún no se instaló el entorno ni se implementó el detector, la API o la interfaz. El dataset sigue pendiente de selección y validación.

## Requisitos académicos

- Dataset limpio y documentado, preferentemente relevante al contexto paraguayo.
- Modelo entrenado con evaluación formal y metodología reproducible.
- API REST para acceder a la predicción.
- Interfaz interactiva con carga de imágenes y visualización de resultados.
- Código Git con desarrollo progresivo, tareas y commits trazables.
- Docker para ejecutar la solución completa (aparece como puntos extra en la consigna).
- Informe IEEE de **exactamente 3 páginas** y demo en vivo en la defensa.

Los requisitos y criterios están detallados en [`docs/01_ALCANCE.md`](docs/01_ALCANCE.md).

## Organización

- [`docs/01_ALCANCE.md`](docs/01_ALCANCE.md): objetivo, alcance y requisitos de la consigna.
- [`docs/02_BACKLOG_INICIAL.md`](docs/02_BACKLOG_INICIAL.md): trabajo propuesto, todavía pendiente de convertir en Issues.
- [`docs/03_DECISIONES_TECNICAS.md`](docs/03_DECISIONES_TECNICAS.md): tecnologías seleccionadas y decisiones que siguen abiertas.
- [`docs/05_ARQUITECTURA_TECNICA.md`](docs/05_ARQUITECTURA_TECNICA.md): componentes, flujo de inferencia, API propuesta y límites.
- [`docs/adr/ADR-001-stack.md`](docs/adr/ADR-001-stack.md): justificación y trade-offs del stack.
- [`docs/04_VALIDACION_Y_RIESGOS.md`](docs/04_VALIDACION_Y_RIESGOS.md): criterios de evaluación y riesgos de datos.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): colaboración de Nicolás y Sebastián.

## Trabajo y contribuciones

En la etapa inicial, **Nicolás registra cambios pequeños y comprobables directamente en `main`**: Issue (cuando corresponda) → implementación → validación → commit → push. No se exigen Pull Requests ni revisiones por ahora. Sebastián será colaborador y registrará contribuciones propias cuando las realice. No se crean commits ficticios para aparentar evolución. Ver [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Fuentes de la consigna

- `Proyecto_ia_para_el_2026.docx`: catálogo, proyecto asignado, entregables y evaluación.
- `Formato presentacion documentos IEEE ES.doc`: formato de referencia para el informe.

Estos documentos se conservan fuera del repositorio hasta decidir cómo incluirlos y bajo qué permisos.
