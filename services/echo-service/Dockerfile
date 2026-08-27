# Run as a normal user (not root) — a bit safer if this container is ever reachable from outside your machine.
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OTEL_SERVICE_NAME=arcs-echo-service \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

COPY services/echo-service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY services/echo-service/main.py .

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
