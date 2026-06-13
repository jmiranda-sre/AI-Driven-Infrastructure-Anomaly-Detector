# Architecture Documentation

## System Context

The AI-Driven Infrastructure Anomaly Detector is a **microservice** that sits between your monitoring infrastructure (Prometheus, Grafana, Kafka) and your alerting/notification systems (Slack, PagerDuty, email). It replaces static threshold-based alerting with ML-powered anomaly detection.

## Data Flow

```
Prometheus ──┐
             │  HTTP/Kafka      Feature          ML             Alert
Kafka ───────┤  ─────────────▶  Engineering ──▶  Pipeline ──▶  Dispatcher
             │   Ingestion      (Rolling Stats)  (Ensemble)     (Multi-Channel)
Custom API ──┘
```

## Component Details

### 1. Ingestion Layer (`src/ingestion/`)

**PrometheusClient**: Async HTTP client for Prometheus API with circuit breaker, timeout, and retry. Supports instant queries and range queries for backfilling.

**KafkaConsumer**: Async Kafka consumer using aiokafka, parses JSON metric messages into MetricPoint objects. Supports batch consumption with auto-commit.

**FeatureEngineer**: Maintains sliding windows per (server_id, metric_name) pair. Computes rolling statistics (mean, std, min, max, skew, kurtosis, rate-of-change) over configurable time windows (5min, 15min, 60min).

**IngestionOrchestrator**: Coordinates all ingestion sources, manages scraping schedules, and dispatches processed features to the ML pipeline via async callback.

### 2. ML Processing Layer (`src/ml_processing/`)

**IsolationForestDetector**: Unsupervised anomaly detection using scikit-learn's IsolationForest. Trained on feature vectors, detects outliers by path length in random forests. O(n·log(n)) training, sub-millisecond inference.

**LSTMAnomalyDetector**: LSTM Autoencoder trained on normal patterns. Reconstruction error above threshold = anomaly. Also includes LSTMForecaster for future value prediction. Requires PyTorch.

**HoltWintersDetector**: Triple exponential smoothing (additive trend + seasonality). Serves as fast, interpretable baseline. Good for univariate metrics with clear daily/weekly seasonality.

**EnsembleDetector**: Combines all model outputs using configurable strategy (weighted average, majority vote, or any-flag). Final anomaly score determines severity.

**DriftDetector**: Monitors concept drift using PSI. Compares reference (training) distribution to current. Triggers retraining when PSI > threshold.

**MLPipeline**: Singleton orchestrator connecting ingestion → models → alerting. Manages training schedules, data buffering, and drift-triggered retraining.

**ModelManager**: Handles model persistence (joblib/torch save/load), versioning, inference metrics recording, and lifecycle management.

### 3. Alerting Layer (`src/alerting/`)

**AlertEvaluator**: Transforms DetectionResult → Alert. Determines severity from score thresholds, applies suppression (cooldown + rate limiting), generates human-readable messages with suggested actions.

**AlertDispatcher**: Multi-channel async dispatcher. Sends to all configured channels concurrently with per-channel circuit breaker protection.

**AlertService**: Facade connecting ML pipeline output to evaluation and dispatch. Maintains alert history, provides query API.

### 4. API Layer (`src/api/`)

FastAPI application with:
- Standard error envelope (DOMAIN_TYPE_DETAIL codes)
- JWT authentication with role-based access control
- Rate limiting (slowapi)
- Security headers middleware
- Request correlation ID propagation
- Prometheus metrics endpoint

### 5. Core Layer (`src/core/`)

Shared infrastructure used by all modules:
- **errors.py**: Hierarchical AppError class with retryable classification
- **config.py**: Layered YAML + env var configuration
- **logging.py**: Structured JSON logging with PII masking (structlog)
- **health.py**: Health check system with dependency verification
- **database.py**: PostgreSQL (asyncpg) + InfluxDB connection management
- **circuit_breaker.py**: Thread-safe circuit breaker for external calls
- **security.py**: JWT token creation/validation, API key hashing

## Security Architecture

```
┌───────────────┐     ┌────────────────┐     ┌──────────────┐
│   Client      │────▶│  TLS Proxy     │────▶│  FastAPI      │
│  (Browser/   │     │  (nginx/       │     │  - JWT Auth   │
│   CLI/API)   │     │   Ingress)     │     │  - Rate Limit │
└───────────────┘     │  - WAF         │     │  - CORS       │
                      │  - HSTS        │     │  - Validation │
                      └────────────────┘     └──────────────┘
```

- **In transit**: TLS 1.3 terminated at Ingress/load balancer
- **At rest**: PostgreSQL passwords, JWT secrets in env vars (→ Vault in production)
- **Authentication**: JWT with HttpOnly cookies (15min access, 7d refresh)
- **Authorization**: Role-based (admin, operator, viewer)
- **Input validation**: Pydantic schemas on every endpoint
- **Rate limiting**: Auth: 5/15min, Default: 100/min

## Resilience Patterns

| Pattern | Implementation | Protection Against |
|---------|---------------|---------------------|
| Circuit Breaker | Per-service breakers (Prometheus, Kafka, Slack, PD) | Cascading failures |
| Retry with Backoff | Exponential backoff on retryable errors | Transient failures |
| Timeout | DB 5s, Internal API 3s, External API 10s | Resource exhaustion |
| Rate Limiting | slowapi (per-IP + per-user) | Abuse, DoS |
| Alert Suppression | Cooldown (30min) + hourly rate cap (50/hr) | Alert storms |
| Dead Letter Queue | Kafka DLQ for failed messages | Data loss |
| Health Checks | /health, /ready, /live with dependency verification | Silent failures |
| Graceful Shutdown | Lifespan cleanup hook | Data corruption |

## Observability Stack

| Pillar | Tool | Format |
|--------|------|--------|
| **Logs** | structlog → JSON stdout → ELK/Loki | Structured JSON with correlation ID |
| **Metrics** | prometheus_client → /metrics | Prometheus exposition format |
| **Tracing** | OpenTelemetry → Jaeger (optional) | W3C Trace Context |

### Key Metrics Exposed

- `http_requests_total{method, endpoint, status}` — Request count
- `http_request_duration_seconds{method, endpoint}` — Latency histogram
- `anomaly_detections_total{server_id, severity}` — Detection count
- `ml_inference_duration_seconds{model_name}` — ML inference latency
- Circuit breaker states, DB/InfluxDB connection pool metrics

## Database Schema

### PostgreSQL (metadata, alerts, models)

```sql
-- Model versions and metadata
models (id, name, version, algorithm, path, metrics, config, created_at, is_active)

-- Configurable alert rules
alert_rules (id, name, metric_pattern, condition_type, threshold, severity, cooldown_min, enabled)

-- Generated alerts (history)
alerts (id, rule_id, server_id, metric_name, anomaly_score, severity, message, metadata, created_at, acknowledged)
```

### InfluxDB (time series features and raw metrics)

- Measurement: `server_metrics`
- Tags: `server_id`, `metric_name`
- Fields: `value`, `anomaly_score`, `is_anomaly`
- Retention: 90d (raw), 1y (downsampled)

## Trade-off Analysis

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Language | Python | Go, Rust | ML ecosystem (scikit-learn, PyTorch) > raw performance |
| Framework | FastAPI | Flask, Django | Async, auto-docs, Pydantic validation |
| ML Primary | Isolation Forest | LOF, OCSVM | No labeled data needed, O(n·log(n)), handles high dimensionality |
| ML Secondary | LSTM Autoencoder | Transformer, VAE | Temporal patterns, forecasting, mature PyTorch support |
| ML Baseline | Holt-Winters | ARIMA | Simpler, handles seasonality natively, very fast |
| Feature Store | In-memory windows | Feast, Redis | Simplicity for single-instance; Redis for distributed |
| Message Queue | Kafka | RabbitMQ, Redis Streams | Kafka for high-throughput, exactly-once, replay |
| Time Series DB | InfluxDB | TimescaleDB, VictoriaMetrics | Purpose-built, Flux query language, good retention |
