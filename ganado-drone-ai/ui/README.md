# Dashboard Streamlit

El dashboard admite hasta 25 fotos por lote, envía cada archivo individualmente
a FastAPI, dibuja las cajas recibidas y muestra conteo y latencia por foto junto
con las detecciones acumuladas. Los errores de una foto no cancelan el resto.

La suma es de detecciones, no de animales únicos: imágenes superpuestas pueden
contar al mismo bovino más de una vez.

```powershell
python -m pip install -r requirements-app.txt
$env:API_URL = 'http://127.0.0.1:8000'
python -m streamlit run ui/app.py
```

En Docker, Compose configura `API_URL=http://api:8000` para la red interna.

La demo muestra el nombre del checkpoint y el umbral informados por `/health`.
El acumulado es una suma de detecciones por fotografía, no una estimación de
animales únicos entre imágenes solapadas.
