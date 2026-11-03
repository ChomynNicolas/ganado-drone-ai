# Servicio de inferencia (pendiente)

Esta carpeta contendrá la API REST definida en `docs/05_ARQUITECTURA_TECNICA.md`.
El paquete Python `api/app/` se creó para alojar el servicio; **aún no hay servidor FastAPI**.

## Contrato previsto

- `GET /health`: estado del proceso y disponibilidad de pesos.
- `POST /predict`: recibe una imagen, valida el archivo y devuelve detecciones.
- Si los pesos no están disponibles, la predicción deberá devolver un error explícito.

La API no calculará métricas de entrenamiento ni será responsable de dibujar las cajas.
La interfaz se comunicará con la API mediante HTTP, sin cargar el modelo por su cuenta.
