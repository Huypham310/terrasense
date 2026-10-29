FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install deps first so code changes don't bust the pip layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# the model is mounted at runtime (see docker-compose.yml), it's too big for the image/repo
ENV TERRASENSE_MODEL_PATH=/app/artifacts/model.onnx \
    TERRASENSE_META_PATH=/app/artifacts/model_meta.json

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
