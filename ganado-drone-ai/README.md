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

**Inicialización del repositorio.** Todavía no hay dataset seleccionado o procesado, modelo entrenado, API funcional ni interfaz implementada. Las decisiones técnicas y resultados experimentales se registrarán conforme se realice el trabajo.

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
- [`docs/03_DECISIONES_PENDIENTES.md`](docs/03_DECISIONES_PENDIENTES.md): decisiones técnicas que deben justificarse.
- [`docs/04_VALIDACION_Y_RIESGOS.md`](docs/04_VALIDACION_Y_RIESGOS.md): criterios de evaluación y riesgos de datos.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): colaboración de Nicolás y Sebastián.

## Trabajo y contribuciones

Las nuevas funcionalidades se incorporan mediante **Issue → rama → commits → Pull Request → revisión → merge**. No se crean commits ficticios para aparentar evolución. Ver [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Fuentes de la consigna

- `Proyecto_ia_para_el_2026.docx`: catálogo, proyecto asignado, entregables y evaluación.
- `Formato presentacion documentos IEEE ES.doc`: formato de referencia para el informe.

Estos documentos se conservan fuera del repositorio hasta decidir cómo incluirlos y bajo qué permisos.
