# 🧠 AI-Driven Infrastructure Anomaly Detector

> Predictive ML-powered anomaly detection for server infrastructure. Move from reactive threshold alerts to intelligent, context-aware anomaly detection.

[![CI/CD](https://github.com/your-org/AI-Driven_Infrastructure_Anomaly_Detector/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/AI-Driven_Infrastructure_Anomaly_Detector/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](deploy/docker/)

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [ML Models & Trade-offs](#-ml-models--trade-offs)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Alert Format](#-alert-format)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [References](#-references)

---

## 🔥 The Problem

Traditional monitoring uses **fixed thresholds** (e.g., "alert if CPU > 80%"). This approach is fundamentally broken:

| Problem | Consequence |
|---------|-------------|
| Same threshold for all servers | False positives on high-utilization servers |
| Ignores historical patterns | Misses slow drift and subtle anomalies |
| No context awareness | Alerts at 3 AM for expected batch processing |
| No prediction | Only reacts *after* impact occurs |
| Alert storms | Fatigue leads to ignoring real alerts |

**Our solution**: Learn each server's *normal behavior* and detect **statistically significant deviations** — including *predicted future anomalies*.

---

## ⚡ How It Works

```
┌──────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌──────────────┐
│  Prometheus  │    │  Feature         │    │  ML Pipeline    │    │  Alerting    │
│  Kafka       │───▶│  Engineering    │───▶│  IF + LSTM + HW │───▶│  Slack       │
│  Custom API  │    │  Rolling Stats  │    │  Ensemble       │    │  PagerDuty   │
└──────────────┘    └─────────────────┘    └────────────────┘    └──────────────┘
      Ingestion          Processing            Detection           Notification
```

1. **Ingest**: Collect metrics from Prometheus, Kafka, or custom sources
2. **Engineer**: Compute rolling statistics (mean, std, skew, rate-of-change) over configurable windows
3. **Detect**: Run ensemble of 3 ML models — Isolation Forest, LSTM Autoencoder, Holt-Winters
4. **Alert**: Evaluate severity, suppress storms, dispatch to Slack/PagerDuty/Webhook/Email
5. **Predict**: Forecast future values and detect *predicted* anomalies before they happen

---

## 🏗️ Architecture

```
                                    ┌────────────────────────────┐
                                    │     FastAPI REST API       │
                                    │  /api/v1/{alerts,models,   │
                                    │   servers,metrics,predict} │
                                    └─────────┬──────────────────┘
                                              │
                   ┌──────────────────────────┼──────────────────────────┐
                   │                          │                          │
          ┌────────▼────────┐     ┌───────────▼────────┐     ┌──────────▼─────────┐
          │  Alert Service    │     │  ML Pipeline        │     │  Model Manager     │
          │  - Evaluator      │     │  - Ensemble         │     │  - Load/Save       │
          │  - Dispatcher     │     │  - Drift Detection   │     │  - Version         │
          │  - Suppression    │     │  - Train/Retrain    │     │  - Metrics         │
          └────────┬─────────┘     └───────────┬─────────┘     └──────────┬─────────┘
                   │                           │                          │
          ┌────────▼───────────────────────────▼──────────────────────────▼─────────┐
          │                        Ingestion Orchestrator                          │
          │  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────────┐     │
          │  │  Prometheus   │  │  Kafka Consumer   │  │  Feature Engineering │     │
          │  │  Client       │  │  (aiokafka)       │  │  (Rolling Windows)   │     │
          │  └───────────────┘  └──────────────────┘  └──────────────────────┘     │
          └─────────────────────────────┬──────────────────────────────────────────┘
                                        │
                   ┌────────────────────┼────────────────────────────┐
                   │                    │                            │
          ┌────────▼──────┐   ┌────────▼───────┐          ┌──────────▼──────────┐
          │  PostgreSQL   │   │  InfluxDB      │          │  Prometheus/Grafana │
          │  (Metadata,   │   │  (Time Series  │          │  (Self-Monitoring)  │
          │   Alerts,     │   │   Features)    │          │                     │
          │   Models)     │   │                │          │                     │
          └───────────────┘   └────────────────┘          └─────────────────────┘
```

### Component Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | ML models | Swap detection algorithms without changing inference code |
| **Observer** | Ingestion → ML → Alerting | Decouple producers from consumers |
| **Circuit Breaker** | External calls (Prometheus, Kafka, Slack) | Prevent cascade failures |
| **Singleton** | ModelManager, ML Pipeline, AlertService | Shared state across requests |
| **Factory** | App factory (create_app) | Clean separation of config and assembly |

---

## 🧮 ML Models & Trade-offs

| Model | Type | Latency | Memory | Handles Seasonality | Predicts Future | Best For |
|-------|------|----------|--------|---------------------|-----------------|----------|
| **Isolation Forest** | Unsupervised | <1ms | Low | ❌ | ❌ | Point anomalies, high-dimensional data |
| **LSTM Autoencoder** | Deep Learning | ~5-50ms | High | ✅ | ✅ | Complex temporal patterns, forecasting |
| **Holt-Winters** | Statistical | <1ms | Very Low | ✅ | ✅ | Univariate series with clear seasonality |

### Ensemble Methods

We combine models using configurable ensemble strategies:

| Method | Recall | Precision | Use Case |
|--------|--------|-----------|----------|
| **Weighted** | Balanced | Balanced | Default — weighted average of scores |
| **Majority** | Lower | Higher | Reduce false positives |
| **Any** | Highest | Lower | Never miss an anomaly |

### Feature Engineering Pipeline

For each (server, metric) pair, we compute rolling statistics over configurable windows:

- **mean**, **std**, **min**, **max** — basic distribution shape
- **skew**, **kurtosis** — detect distribution shape changes
- **rate of change** — detect sudden spikes/drops

Windows: 5min, 15min, 60min (configurable)

### Drift Detection

Uses **Population Stability Index (PSI)** to detect concept drift:

| PSI Range | Drift Level | Action |
|-----------|-------------|--------|
| PSI < 0.10 | No drift | Continue |
| 0.10 ≤ PSI < 0.25 | Moderate drift | Log warning |
| PSI ≥ 0.25 | Significant drift | Trigger retraining |

---

## 🚀 Quick Start

### Docker Compose (recommended for local development)

```bash
# Clone
git clone https://github.com/your-org/AI-Driven_Infrastructure_Anomaly_Detector.git
cd AI-Driven_Infrastructure_Anomaly_Detector

# Configure
cp .env.example .env
# Edit .env with your secrets

# Start all services
docker compose -f deploy/docker/docker-compose.yml up -d

# Generate sample data and train models
docker compose -f deploy/docker/docker-compose.yml exec anomaly-detector \
    python -m data.scripts.generate_sample
docker compose -f deploy/docker/docker-compose.yml exec anomaly-detector \
    python -m models.scripts.train --source synthetic --samples 5000

# Access
# API: http://localhost:8000/docs
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### Standalone (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,full]"

# Configure
cp .env.example .env

# Generate training data
python -m data.scripts.generate_sample

# Train models
python -m models.scripts.train --source synthetic --samples 5000

# Start API
python -m src.main

# Run tests
pytest tests/ -v
```

---

## ⚙️ Configuration

### Layered Configuration

Precedence (highest → lowest):
1. **Environment variables** (`SERVICE_PORT`, `POSTGRES_HOST`, `JWT_SECRET`, etc.)
2. **Environment-specific YAML** (`config/production.yaml`, `config/test.yaml`)
3. **Default YAML** (`config/default.yaml`)
4. **AD__ prefix override** (`AD__ML__MODELS__ISOLATION_FOREST__CONTAMINATION=0.02`)

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_PORT` | 8000 | API server port |
| `SERVICE_ENV` | development | Environment |
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PASSWORD` | — | PostgreSQL password |
| `PROMETHEUS_URL` | http://localhost:9090 | Prometheus endpoint |
| `JWT_SECRET` | — | JWT signing key (required in prod) |
| `SLACK_WEBHOOK_URL` | — | Slack webhook for alerts |
| `PAGERDUTY_ROUTING_KEY` | — | PagerDuty routing key |

See [`.env.example`](.env.example) for the full list.

---

## 📡 API Reference

Base URL: `/api/v1/`

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health with dependency checks |
| `/ready` | GET | Kubernetes readiness probe |
| `/live` | GET | Kubernetes liveness probe |
| `/metrics` | GET | Prometheus metrics scrape |

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | Get JWT access + refresh tokens |

### Alerts

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/alerts` | GET | List recent alerts (paginated) |
| `/api/v1/alerts/{id}` | GET | Get specific alert |
| `/api/v1/alerts/{id}/acknowledge` | POST | Acknowledge alert |
| `/api/v1/alerts/stats/summary` | GET | Alert statistics |

### ML Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/models` | GET | List loaded models and status |
| `/api/v1/models/train` | POST | Trigger model training |
| `/api/v1/models/drift` | GET | Check model drift status |

### Servers & Metrics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/servers` | GET | List monitored servers |
| `/api/v1/servers/{id}` | GET | Server detail with metrics |
| `/api/v1/metrics/latest` | GET | Latest raw metrics |
| `/api/v1/predictions` | POST | Get anomaly prediction |

All responses follow the standard envelope:

```json
{ "data": { ... } }
```

Errors follow the `DOMAIN_TYPE_DETAIL` code system:

```json
{
  "error": {
    "code": "AUTH_INVALID_CREDENTIALS",
    "message": "Invalid email or password",
    "request_id": "req_abc123",
    "retryable": false
  }
}
```

---

## 🔔 Alert Format

Example structured alert JSON:

```json
{
  "alert_id": "alt_a1b2c3d4e5f6",
  "server_id": "prod-web-01",
  "metric_name": "cpu_usage",
  "anomaly_score": 0.87,
  "severity": "critical",
  "is_predicted": false,
  "prediction_horizon": null,
  "message": "CRITICAL: cpu_usage on prod-web-01 at 97.3% deviates from predicted normal of 48.2% (anomaly score: 0.87)",
  "suggested_action": "Investigate high CPU usage — check for runaway processes, consider scaling",
  "model_details": {
    "method": "weighted",
    "models": {
      "isolation_forest": {"score": 0.91, "is_anomaly": true, "weight": 0.4},
      "lstm_autoencoder": {
        "score": 0.85,
        "is_anomaly": true,
        "reconstruction_error": 0.042,
        "weight": 0.4
      },
      "holt_winters": {
        "score": 0.79,
        "is_anomaly": true,
        "forecast": 48.2,
        "z_score": 3.4,
        "weight": 0.2
      }
    }
  },
  "dashboard_url": "http://grafana:3000/d/server-detail?var-server=prod-web-01",
  "timestamp": "2026-06-13T10:30:00Z",
  "correlation_id": "req_abc123"
}
```

---

## 🐳 Deployment

### Docker

```bash
docker build -f deploy/docker/Dockerfile -t anomaly-detector:latest .
docker run -p 8000:8000 --env-file .env anomaly-detector:latest
```

### Kubernetes

```bash
kubectl apply -f deploy/kubernetes/
```

### Helm

```bash
helm install anomaly-detector deploy/helm/anomaly-detector/ \
  --set secrets.postgresPassword=$PG_PASS \
  --set secrets.jwtSecret=$JWT_SECRET
```

### Production Checklist

- [ ] Set `SERVICE_ENV=production`
- [ ] Set strong `JWT_SECRET` (≥64 chars random)
- [ ] Set `POSTGRES_PASSWORD`
- [ ] Set `INFLUXDB_TOKEN`
- [ ] Configure alert channels (Slack, PagerDuty)
- [ ] Enable TLS (Ingress or load balancer)
- [ ] Set CORS origins to production domains
- [ ] Configure resource limits (1GB mem, 1 CPU default)
- [ ] Enable HPA (min 2, max 8 replicas)
- [ ] Set up backup for PostgreSQL and InfluxDB
- [ ] Configure model retraining schedule

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires services)
pytest tests/integration -v

# Performance benchmarks
pytest tests/performance -v -m performance

# With coverage
pytest tests/ --cov=src --cov-report=html

# Security scan
pip-audit
bandit -r src/ -ll
```

### ML Model Validation

| Strategy | Description |
|----------|-------------|
| **Backtesting** | Test model on historical data with known anomalies |
| **Shadow mode** | Run model alongside threshold alerts, compare results |
| **A/B testing** | Route subset of traffic to new model version |
| **PSI monitoring** | Track drift score, alert on significant drift |
| **Champion/Challenger** | Deploy new model alongside current, compare metrics |

---

## 🗺️ Roadmap

### MVP (v0.1) — ✅ Current
- [x] Prometheus ingestion
- [x] Isolation Forest detector
- [x] Holt-Winters detector
- [x] Ensemble detection
- [x] Webhook alerts
- [x] REST API
- [x] Docker deployment
- [x] Health checks & metrics

### v1.0 — Production Ready
- [ ] LSTM Autoencoder (requires PyTorch)
- [ ] Kafka ingestion
- [ ] Slack + PagerDuty + Email channels
- [ ] Drift detection with auto-retraining
- [ ] JWT authentication
- [ ] Kubernetes Helm chart
- [ ] Grafana dashboards
- [ ] Comprehensive test suite (>80% coverage)

### v2.0 — Advanced Features
- [ ] DBSCAN clustering model
- [ ] One-Class SVM model
- [ ] Multi-variate cross-correlation detection
- [ ] Causal analysis (which metric caused the anomaly)
- [ ] Auto-remediation hooks (scale up, restart)
- [ ] Federation (multi-cluster monitoring)
- [ ] gRPC API alongside REST
- [ ] Model marketplace (community-contributed models)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, development setup, and architecture overview.

---

## 📚 References

1. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). *Isolation Forest*. IEEE ICDM. — Foundation of the Isolation Forest algorithm, demonstrating efficient anomaly isolation through random partitioning.

2. Malhotra, P., et al. (2015). *Long Short Term Memory Networks for Anomaly Detection in Time Series*. ESANN. — LSTM-based approach for temporal anomaly detection using reconstruction error.

3. Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys. — Comprehensive survey of anomaly detection techniques.

4. Winters, P.R. (1960). *Forecasting Sales by Exponentially Weighted Moving Averages*. Management Science. — Foundation of Holt-Winters exponential smoothing.

5. Box, G.E.P., & Draper, N.R. (1987). *Empirical Model-Building and Response Surfaces*. Wiley. — Statistical foundations for model evaluation.

6. Population Stability Index (PSI): [FDIC Supervisory Insights](https://www.fdic.gov/bank examinations/) — Industry standard for detecting distribution shifts.

7. Google SRE Book (2016). *Site Reliability Engineering*. O'Reilly. — SLI/SLO framework and alerting best practices.

8. [Prometheus Metrics Best Practices](https://prometheus.io/docs/practices/) — Instrumentation and metric naming conventions.

9. [OWASP Top 10 (2021)](https://owasp.org/Top10/) — Web application security checklist.

10. [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/) — Production patterns for Python APIs.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Built with ❤️ for SREs, DevOps engineers, and anyone tired of false-positive threshold alerts.**
