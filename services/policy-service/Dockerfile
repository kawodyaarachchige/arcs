FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Install the shared library (arcs_rl + gRPC stubs under arcs.policy.v1) from the repo root metadata.
COPY pyproject.toml README.md LICENSE ./
COPY arcs-rl arcs-rl
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[train]"

COPY services/policy-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/policy-service/*.py .

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080
EXPOSE 50051

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
