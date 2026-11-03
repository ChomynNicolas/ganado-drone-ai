# Validación experimental y riesgos

## Evaluación mínima esperada

1. Documentar fuente del dataset, versión, licencia, clases, imágenes y anotaciones.
2. Definir particiones train/validation/test antes de entrenar; si las imágenes provienen de la misma secuencia de vuelo, agrupar por vuelo o escena cuando sea posible para evitar fuga de información.
3. Registrar semillas, versiones, hiperparámetros, resolución, pesos iniciales y hardware.
4. Separar baseline preentrenado de modelo ajustado: ambos deben evaluarse bajo el mismo protocolo.
5. Informar mAP@0.5 y mAP@0.5:0.95, precisión y recall del detector; agregar MAE de conteo por imagen e inferencia en ms/imagen (con hardware identificado).
6. Mostrar errores en escenas densas, animales muy pequeños, sombras, solapamientos y falsos positivos.

No anunciar un valor de métricas hasta ejecutar evaluaciones reales.

## Riesgos críticos

| Riesgo | Efecto | Acción preventiva |
|---|---|---|
| Dataset no disponible o licencia restrictiva | bloqueo de entrenamiento o publicación | validar fuente/licencia antes de elegir |
| Leakage entre imágenes casi idénticas | métricas optimistas | split agrupado; inspección de duplicados |
| Animales muy pequeños | bajo recall | resolución adecuada, tiling si es necesario |
| Fotos superpuestas del lote | doble conteo | total etiquetado como detecciones, no cabezas únicas |
| GPU insuficiente | entrenamiento falla | batch pequeño, modelo pequeño o Colab |
| Demo dependiente de Internet | falla durante defensa | ejecución local y fotografías probadas |
| Pesos demasiado grandes para Git | repo pesado/bloqueos | usar artefacto de release o almacenamiento externo documentado |
