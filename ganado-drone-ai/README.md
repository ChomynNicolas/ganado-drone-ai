# Ganado Drone AI

**Proyecto 2 — Conteo y Monitoreo de Ganado Vacuno mediante Fotografías Aéreas de Drones**

Proyecto académico de Inteligencia Artificial (4.º año, Ingeniería Informática, Paraguay).

**Integrantes:** Nicolás Chomynn y Sebastián Paredes.

## Problema

Diseñar un sistema de visión por computadora que detecte y contabilice ganado bovino en fotografías aéreas tomadas por drones. El objetivo es apoyar tareas de inventario de animales en establecimientos ganaderos.

## Resultado esperado

Una aplicación que permita subir un lote de imágenes aéreas y visualizar las detecciones con cajas delimitadoras, cantidad por imagen y un total acumulado de detecciones del lote.

> Si las fotografías se superponen, una vaca puede aparecer en varias imágenes. La suma de detecciones no constituye un censo de animales únicos.

## Estado actual

**Stack definido, entorno Python y validadores de datos implementados.** Se seleccionó WAID como fuente candidata y se agregó el auditor de su formato original. **No se ha descargado ni auditado localmente WAID**, convertido las etiquetas, entrenado un detector ni construido la API/UI.

## Desarrollo local

**Requisito:** Python 3.11 instalado.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check scripts tests
.\.venv\Scripts\python.exe -m ruff format --check scripts tests
```

## Dataset candidato: WAID

Fuente oficial: https://github.com/xiaohuicui/WAID

La fuente original dispone de imágenes/etiquetas en `train`, `valid` y `test` y `classes.txt` declara `cattle=1`. **No usar el validador de una clase sobre etiquetas originales.**

Descarga, auditoría, licencia y reproducibilidad: [`docs/06_FUENTE_WAID.md`](docs/06_FUENTE_WAID.md).

```powershell
git clone --depth 1 --filter=blob:none https://github.com/xiaohuicui/WAID.git data/raw/waid-repo
.\.venv\Scripts\python.exe -m scripts.audit_waid --dataset data/raw/waid-repo/WAID --hash-check --output reports/generated/waid_audit.json
```

Los datasets descargados quedan en `data/raw/`, ignorado por Git. Las estadísticas reales deben obtenerse de la auditoría; no se atribuyen valores hipotéticos a entrenamientos todavía no realizados.

## Estructura y documentación

- `api/app/`: backend aún no implementado.
- `ui/`: dashboard aún no implementado.
- `data/DATASET_CARD.md`: fuente, etiquetas, restricciones y evidencia pendiente.
- `configs/`: futura configuración de entrenamiento.
- `models/`: checkpoints locales ignorados por Git.
- `reports/`: informe y métricas; `reports/generated/` está ignorado.
- `scripts/`: herramientas de control del entorno y datos.
- `tests/`: pruebas sintéticas de scripts; **no** validación real del modelo.
- [`docs/01_ALCANCE.md`](docs/01_ALCANCE.md): consigna y alcance.
- [`docs/02_BACKLOG_INICIAL.md`](docs/02_BACKLOG_INICIAL.md): actividades y criterios de aceptación.
- [`docs/03_DECISIONES_TECNICAS.md`](docs/03_DECISIONES_TECNICAS.md): selección de tecnologías.
- [`docs/05_ARQUITECTURA_TECNICA.md`](docs/05_ARQUITECTURA_TECNICA.md): componentes previstos.
- [`docs/06_FUENTE_WAID.md`](docs/06_FUENTE_WAID.md): fuente de imágenes y adquisición.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): historial Git y colaboración.

## Trabajo en Git

Nicolás registra cambios pequeños comprobados en `main` y Sebastián utiliza su cuenta e identidad Git cuando contribuya. Las imágenes, las etiquetas y los pesos entrenados se mantienen fuera del repositorio. Los Issues pueden documentar cambios sin PR obligatorios en esta etapa.

## Entregables pendientes

Datos procesados y evaluados, detector con fine-tuning reproducible, API REST, interfaz para lotes, Docker Compose, informe IEEE de exactamente tres páginas y demostración en vivo. Nada de esto debe declararse como terminado hasta que haya evidencia de ejecución.
