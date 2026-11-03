# Guía de contribución — Ganado Drone AI

## Estado del flujo

Durante el arranque del proyecto, **Nicolás realiza los commits y publica directamente en `main`**, sin Pull Requests ni revisión obligatoria. Es una **decisión operativa temporal**, no una regla académica ni una indicación de que Sebastián ya desarrolló código.

Sebastián dispone de acceso de colaborador para consultar Issues, documentación y resultados. Cuando comience a implementar, debe configurar su propia identidad de Git y registrar sus contribuciones reales; se reconsiderará el uso de ramas y PR si ambos trabajan en paralelo.

## Proceso por cambio

1. Crear o actualizar un **Issue** con objetivo y criterios de aceptación, cuando el cambio sea una tarea sustantiva o un bug. La inicialización documental no requiere Issues retroactivos.
2. Sincronizar `main` con `git pull --ff-only origin main` **antes** de comenzar.
3. Hacer un cambio acotado a un objetivo: por ejemplo configuración, test, preparación de datos, endpoint o corrección.
4. Ejecutar una comprobación pertinente: test, comando, validación o inspección documental.
5. Revisar `git diff`, incluir solo archivos relacionados (`git add <archivos>`) y verificar `git diff --cached --check`.
6. Registrar un commit descriptivo y publicar con `git push origin main`.
7. Cerrar el Issue solamente si sus criterios se cumplen y agregar evidencia (comando y resultado, captura o archivo) cuando aplique.

## Comandos habituales (PowerShell)

```powershell
git switch main
git pull --ff-only origin main
git status --short
# Editar/crear los archivos de la tarea
# Ejecutar las pruebas adecuadas
git diff --check
git diff
git add RUTA_1 RUTA_2
git diff --cached --check
git diff --cached --stat
git commit -m "tipo(area): short description"
git push origin main
git status
```

**No ejecutar `git add .` de forma automática** sin revisar archivos; no incluir `.venv`, `.env`, datasets, pesos, datos privados ni artefactos grandes. No usar `git push --force` en `main`.

## Formato de commits

`tipo(area): descripción breve en inglés`.

Tipos: `feat` (capacidad nueva), `fix` (error corregido), `test` (pruebas), `docs` (documentación), `chore` (configuración/mantenimiento), `refactor` (cambio interno sin modificar comportamiento), `ci` (automatización CI).

Ejemplos **solo si reflejan cambios implementados**:

- `docs(process): document direct-to-main workflow`
- `chore(env): add Python version and dependency files`
- `feat(data): convert cattle labels to YOLO format`
- `test(api): reject invalid image uploads`
- `fix(inference): preserve source image dimensions`

Si el commit completa un Issue real de GitHub, puede relacionarse en el mensaje o en el Issue con `Closes #N` (usando el número real, no el código local IA-xx). Para incidencias parciales, usar `Refs #N` sin cerrarlo.

## Evidencia y trazabilidad

- La historia de Git debe reflejar **qué se cambió realmente**, no simular trabajo pasado.
- Documentar dataset y licencia, splits sin fuga entre escenas, comandos de entrenamiento, métricas y versiones según se produzcan.
- Guardar decisiones de arquitectura en `docs/`; no declarar como implementada una tecnología que solo está propuesta.
- Cuando aparezca un error, documentar reproducción, causa y solución; añadir test de regresión cuando sea posible.
- Antes de cada push, verificar que los archivos afectados son los esperados y que no hay secretos.

## Trabajo paralelo en el futuro

Si ambos modifican código de forma simultánea, pasar a ramas por tarea y Pull Requests, con o sin revisión formal según la disponibilidad. Evitar que ambos hagan commits directamente sobre `main` al mismo tiempo.
