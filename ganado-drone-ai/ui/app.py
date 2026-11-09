"""Streamlit dashboard that calls the FastAPI detector over HTTP."""

from __future__ import annotations

import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_UPLOAD_MB = 20
MAX_BATCH_FILES = 25

st.set_page_config(page_title="Ganado Drone AI", page_icon="🐄", layout="wide")
st.title("Ganado Drone AI")
st.caption("Detección de bovinos por fotografía aérea")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
except requests.RequestException:
    health = None

if not health:
    st.error("No se puede conectar con la API. Comprueba que el servicio esté iniciado.")
elif health.get("model_loaded"):
    confidence = health.get("confidence")
    confidence_text = f" · umbral: {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
    checkpoint = health.get("checkpoint") or "desconocido"
    st.success(
        f"Detector listo · modelo: {checkpoint} · dispositivo: "
        f"{health.get('device', 'desconocido')}{confidence_text}"
    )
else:
    st.warning("La API responde, pero no tiene un checkpoint disponible. Configura MODEL_WEIGHTS.")

uploads = st.file_uploader(
    "Selecciona una o más fotografías",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    max_upload_size=MAX_UPLOAD_MB,
    help=f"JPG o PNG; máximo {MAX_UPLOAD_MB} MiB por fotografía.",
)
if uploads and len(uploads) > MAX_BATCH_FILES:
    st.error(f"El lote admite como máximo {MAX_BATCH_FILES} imágenes.")


def draw_detections(image_bytes: bytes, detections: list[dict]) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 400))
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline="#ff4b4b", width=line_width)
        label = f"cattle {float(detection['confidence']):.2f}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill="#ff4b4b")
        draw.text((x1, y1), label, fill="white", font=font)
    return image


def analyze_file(uploaded_file) -> dict:
    payload = uploaded_file.getvalue()
    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                uploaded_file.name,
                payload,
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=(5, 180),
    )
    response.raise_for_status()
    result = response.json()
    if result.get("count") != len(result.get("detections", [])):
        raise ValueError("La respuesta de API no coincide entre count y detections")
    result["annotated"] = draw_detections(payload, result["detections"])
    return result


if st.button(
    "Analizar imágenes",
    type="primary",
    disabled=(
        not uploads
        or len(uploads) > MAX_BATCH_FILES
        or not health
        or not health.get("model_loaded")
    ),
):
    results: list[dict] = []
    progress = st.progress(0, text="Enviando imágenes a la API...")
    for index, uploaded_file in enumerate(uploads, start=1):
        try:
            result = analyze_file(uploaded_file)
            results.append({"ok": True, "data": result})
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            results.append({"ok": False, "name": uploaded_file.name, "error": detail})
        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            results.append({"ok": False, "name": uploaded_file.name, "error": str(exc)})
        progress.progress(index / len(uploads), text=f"Procesadas {index} de {len(uploads)}")
    st.session_state["analysis_results"] = results

results = st.session_state.get("analysis_results", [])
successful = [item["data"] for item in results if item.get("ok")]
if successful:
    st.divider()
    total = sum(result["count"] for result in successful)
    st.metric("Detecciones acumuladas del lote", total)
    st.info(
        "La suma cuenta detecciones por fotografía. Si dos fotos se superponen, "
        "un mismo bovino puede contarse más de una vez."
    )
    columns = st.columns(2)
    for index, result in enumerate(successful):
        with columns[index % 2]:
            st.subheader(result["image_name"])
            st.image(result["annotated"], caption=f"{result['count']} detecciones", width="stretch")
            st.caption(
                f"{result['width']}×{result['height']} px · "
                f"{result['inference_ms']:.1f} ms de inferencia"
            )

for item in results:
    if not item.get("ok"):
        st.error(f"{item['name']}: {item['error']}")
