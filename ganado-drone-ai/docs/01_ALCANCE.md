# Alcance y consigna — Proyecto 2

## Origen

Documento base: `Proyecto_ia_para_el_2026.docx`, sección «Proyecto 2: Conteo y Monitoreo de Ganado Vacuno mediante Fotografías Aéreas de Drones» y directivas generales de evaluación.

## Problema

El conteo manual del ganado en potreros es costoso en tiempo y susceptible a errores. El proyecto propone evaluar detección de objetos sobre fotografías aéreas para ubicar y contar bovinos.

## Objetivo general

Desarrollar y evaluar un pipeline de visión por computadora capaz de localizar, delimitar y contar ganado bovino en imágenes aéreas.

## Resultado esperado del proyecto específico

Dashboard interactivo para subir un **lote de imágenes de dron**, presentar detecciones con **bounding boxes** y mostrar la cantidad detectada por imagen y total acumulado del lote.

## Entregables del catálogo (no opcionales, salvo donde se indique)

1. Dataset limpio y documentado; adaptación paraguaya **preferida**, no garantizada por la consigna.
2. Modelo entrenado y evaluado mediante métricas formales adecuadas al detector.
3. API REST funcional.
4. Interfaz interactiva.
5. Código Git e historial de desarrollo con historias de usuario.
6. Docker; el bloque de entregables identifica `Dockerfile` y `docker-compose.yml` como **puntos extra**.
7. Informe científico IEEE de dos columnas, **exactamente 3 páginas**, con Abstract (máximo 150 palabras), introducción, metodología, resultados/discusión, conclusiones/trabajo futuro y referencias.
8. Defensa: 10 minutos de exposición, 2 minutos de preguntas y demo en vivo.

**Rúbrica:** ingeniería 30 puntos, defensa y demo 50 puntos, informe IEEE 20 puntos.

## Alcance de ingeniería propuesto (a validar)

**Incluido:** detección de una clase `cattle`, conteo en fotografía estática, lote de imágenes, anotaciones visuales, evaluación de precisión y error de conteo, inferencia local reproducible.

**Fuera de alcance inicial:** identificación individual de animales, georreferenciación exacta, deduplicación entre imágenes, seguimiento en video, control de drones, registro de usuarios y facturación.

Estas exclusiones **son decisiones propuestas del equipo**, no requisitos del profesor.

## Criterios de aceptación del producto final

- Una persona puede cargar por lo menos dos imágenes y observar predicciones con cajas delimitadoras.
- El conteo por fotografía coincide con la cantidad de cajas válidas presentada.
- Se muestra la suma de **detecciones** del lote, explicando que no necesariamente equivale a animales únicos.
- Las métricas del modelo se calculan sobre un conjunto de test separado de entrenamiento y selección de hiperparámetros.
- El sistema funciona sin conexión a servicios externos durante la demo, o existe un plan B documentado.

## Por confirmar

- Dataset concreto, condiciones de licencia y procedencia.
- Arquitectura de detector y herramientas de interfaz.
- Disponibilidad real de imágenes paraguayas.
- Hardware y tiempos de entrenamiento observados.
