# Builds a small HTTP service that turns Prometheus scrapes and optional Kafka events into the 12-number state vector.
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY arcs-rl arcs-rl
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "."

COPY services/telemetry-bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/telemetry-bridge/*.py .

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8085
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085"]
