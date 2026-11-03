# Colaboración — Nicolás y Sebastián

## Modelo de trabajo

1. Crear un Issue con criterios de aceptación verificables.
2. Asignar autor y revisor: quien implementa no aprueba su propio PR.
3. Actualizar `main`: `git switch main` y `git pull --ff-only origin main`.
4. Crear rama corta: `feat/ia-03-dataset`, `fix/ia-07-labels` o `docs/ia-01-stack`.
5. Realizar cambios de alcance reducido y validar localmente.
6. Commit atómico con mensaje representativo.
7. Hacer push de la rama y abrir PR con `Closes #N` si realmente completa el Issue.
8. Revisor ejecuta verificaciones, deja comentario y aprueba. Autor corrige si hace falta.
9. Fusionar el PR y eliminar la rama remota; volver a actualizar `main`.

## Formato de commits

`tipo(área): descripción en inglés`

Tipos sugeridos: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`.

Ejemplos (solo cuando describan cambios reales):

- `docs(scope): clarify detection and counting requirements`
- `chore(env): add reproducible Python setup`
- `feat(data): validate cattle bounding box annotations`
- `test(api): cover invalid image uploads`
- `fix(inference): correct bounding box coordinate scaling`

Evitar: `update`, `fix`, `avance`, `final`, `prueba123`.

## Reglas

- `main` representa una versión consistente. No trabajar directamente sobre ella después del primer commit.
- No usar `git push --force` en `main`.
- No alterar fechas de commits ni crear commits vacíos para aparentar trabajo.
- No subir datos crudos, pesos grandes, claves, `.env`, capturas privadas o información de terceros sin permiso.
- No confundir una plantilla de documentación con funcionalidad implementada.
- Si un PR no pasa sus criterios, se corrige con un nuevo commit real.
- Si el repositorio privado no dispone de branch protection por el plan, mantener revisión por PR como política del equipo, sin afirmar que está técnicamente obligada.

## Definición de terminado

- Issue enlazado y alcance satisfecho.
- Pruebas o evidencias que corresponden a esa tarea.
- README/documentación actualizados si cambió uso o arquitectura.
- PR revisado por la otra persona y fusionado.
