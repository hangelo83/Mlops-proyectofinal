# Dockerfile — Imagen de la API de prevención de infartos
# Usado por GitHub Actions (job build-docker) y por make docker-build.
# Requiere haber corrido el pipeline antes (python main.py) para tener artifacts/.

# ── Imagen base: Python 3.11 slim (misma versión que el Codespace y CI) ──────
FROM python:3.11-slim

WORKDIR /app

# ── Dependencias Python — copiar ANTES del código para aprovechar el cache ────
# Estas instrucciones son idénticas a las de Dockerfile.trainer: Docker construye
# la capa de pip install UNA sola vez y la comparte entre las dos imágenes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

LABEL maintainer="MLOps Prevención de Infartos"
LABEL description="API de predicción de riesgo de infarto"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENV=dev

# ── Código de la API + transformaciones compartidas con el pipeline ───────────
COPY api/ ./api/
COPY src/ ./src/

# ── Artefactos del pipeline (modelo + parámetros de preprocesamiento) ─────────
COPY artifacts/modelo_infartos.pkl     ./artifacts/
COPY artifacts/preprocesamiento.json   ./artifacts/
COPY artifacts/eval_metrics.json       ./artifacts/

EXPOSE 8000

# ── Health check para que Docker sepa si el contenedor está sano ──────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
