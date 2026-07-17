# Single image, reused by both the "api" and "mlflow" services in
# docker-compose.yml (see command overrides there). Self-contained: the
# model is trained as part of the image build, so the image alone is
# enough to serve predictions -- no external volume or init step required.
FROM python:3.10-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by xgboost at import time on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY data/ data/

# Bakes models/best_model.joblib, models/best_model_metadata.json,
# models/shap_background.joblib, and mlflow.db (+ mlruns/) into the image.
RUN python -m src.train

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
