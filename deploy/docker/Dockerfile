# ── Stage 1: Build ──────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

# Build deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/
COPY config/ config/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -e ".[full]"

# ── Stage 2: Production ─────────────────────────────────────────
FROM python:3.14-slim AS production

# Runtime deps only (libpq for asyncpg, no gcc)
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r anomaly && useradd -r -g anomaly -d /app -s /sbin/nologin anomaly

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /app /app

# Security: no bytecode, no buffer, production env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_ENV=production \
    PATH="/usr/local/bin:${PATH}"

# Create necessary directories with proper permissions
RUN mkdir -p /app/models/trained && \
    chown -R anomaly:anomaly /app

USER anomaly

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.main"]
