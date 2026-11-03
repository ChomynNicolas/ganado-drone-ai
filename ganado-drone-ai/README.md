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

**Arquitectura y entorno documentados; primera herramienta de control de datos disponible.** Se puede validar la geometría de etiquetas YOLO de una clase, pero todavía **no** se ha descargado un dataset real, entrenado un detector ni implementado la API o UI.

## Requisitos académicos

- Dataset limpio y documentado, preferentemente relevante al contexto paraguayo.
- Modelo entrenado con evaluación formal y metodología reproducible.
- API REST para acceder a la predicción.
- Interfaz interactiva con carga de imágenes y visualización de resultados.
- Código Git con desarrollo progresivo, tareas y commits trazables.
- Docker para ejecutar la solución completa (aparece como puntos extra en la consigna).
- Informe IEEE de **exactamente 3 páginas** y demo en vivo en la defensa.

Los requisitos y criterios están detallados en [`docs/01_ALCANCE.md`](docs/01_ALCANCE.md).


## Entorno de desarrollo (Windows / PowerShell)

**Requisito:** Python 3.11 instalado (comprobar con `py -3.11 --version`). El entorno virtual y sus paquetes no se suben a Git.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check scripts tests
.\.venv\Scripts\python.exe -m ruff format --check scripts tests
```

Si el comando `py -3.11` falla, instalar primero Python 3.11; no reemplazarlo silenciosamente por otra versión. Los comandos usan el intérprete de `.venv` directamente y no requieren activar el entorno ni cambiar políticas de PowerShell.

## Validación de etiquetas YOLO (sin dataset descargado)

Después de convertir las anotaciones originales a la **única clase `cattle=0`**,
ejecutar el validador sobre un archivo o directorio local:

```powershell
.\.venv\Scripts\python.exe scripts/validate_yolo_labels.py --labels data/processed/labels/train
```

El comando verifica formato de cinco columnas, ID de clase, rangos normalizados,
cajas dentro de la imagen, valores no finitos y etiquetas idénticas duplicadas.
Devuelve código **0** sin errores, **1** si hay anotaciones inválidas o **2** si los
argumentos/ruta son incorrectos. No descarga datos, no repara etiquetas y **no**
verifica correspondencia imagen-etiqueta, calidad visual ni leakage entre splits.
El script solo se ejecutará con datos reales cuando existan.

`requirements-dev.txt` fija por ahora **solo** `pytest` y `ruff`; las dependencias del entrenamiento, API y UI se incorporarán cuando exista código que las necesite. `pyproject.toml` configura las herramientas, **no** declara aún un paquete Python instalable. La verificación prueba la configuración de desarrollo, no el desempeño del futuro modelo.

## Estructura de trabajo

- `api/app/`: paquete reservado para el backend; no existe endpoint operativo.
- `ui/`: interfaz prevista, aún sin código ejecutable.
- `data/DATASET_CARD.md`: ficha para documentar los datos cuando se adquieran.
- `configs/`: configuraciones versionadas de experimentos (sin rutas locales privadas).
- `models/`: checkpoints locales excluidos de Git; se documentará cómo obtenerlos.
- `reports/`: material de métricas y del informe científico.
- `scripts/`: verificación del entorno y validación estática de etiquetas YOLO; entrenamiento pendiente.
- `tests/`: pruebas automatizadas de las funcionalidades existentes.

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
