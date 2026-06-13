# ── Stage 1: Build ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
COPY models/ models/
COPY config/ config/
COPY data/ data/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[full]"

# ── Stage 2: Production ─────────────────────────────────────────
FROM python:3.12-slim AS production

# Security: non-root user
RUN groupadd -r anomaly && useradd -r -g anomaly -d /app -s /sbin/nologin anomaly

WORKDIR /app

# Install runtime dependencies only
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Security headers, healthcheck, resource limits
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_ENV=production

# Create necessary directories with proper permissions
RUN mkdir -p /app/models/trained && \
    chown -R anomaly:anomaly /app

USER anomaly

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"] || exit 1

CMD ["python", "-m", "src.main"]
