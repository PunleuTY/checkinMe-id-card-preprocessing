FROM python:3.12-slim

# opencv-python-headless still needs libglib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching. gunicorn is a deploy-only
# concern, so it lives here rather than in requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Cloud Run injects $PORT (default 8080). Shell form so $PORT expands.
# 1 worker + threads: the prod path /gemini-ocr/upload is IO-bound (waiting on
# the Gemini API, released via run_in_threadpool), so threads handle concurrency
# cheaply. --timeout 120 keeps slow gemini-2.5-pro calls from killing the worker.
CMD exec gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --bind 0.0.0.0:${PORT:-8080}
