FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY docker/dashboard-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/common/ src/common/
COPY dashboard/api/ dashboard/api/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
