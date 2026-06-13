# Operational Playbook

## Runbook: Anomaly Detector Operations

### Startup Procedure

1. Verify all dependencies are healthy: PostgreSQL, InfluxDB, Prometheus
2. Start service: `docker compose up -d` or `python -m src.main`
3. Check `/health` returns `healthy`
4. Verify Prometheus scrape at `/metrics`
5. Confirm models loaded: `GET /api/v1/models`

### Common Operations

#### Manual Model Retraining
```bash
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Authorization: Bearer $TOKEN"
```

#### Check Drift Status
```bash
curl http://localhost:8000/api/v1/models/drift \
  -H "Authorization: Bearer $TOKEN"
```

#### View Recent Alerts
```bash
curl http://localhost:8000/api/v1/alerts?severity=critical&limit=10 \
  -H "Authorization: Bearer $TOKEN"
```

### Troubleshooting

#### High False Positive Rate
1. Check anomaly score distribution: `GET /api/v1/alerts`
2. Raise threshold in config: `ml.inference.threshold`
3. Switch ensemble method from `any` to `weighted`
4. Increase `alerting.suppression.cooldown_minutes`

#### Model Drift Detected
1. Check drift report: `GET /api/v1/models/drift`
2. If PSI > 0.25, trigger retraining
3. If retraining fails, fall back to previous model version
4. Investigate root cause: data source change? new workload pattern?

#### Circuit Breaker Open
1. Identify which breaker: check `/health` details
2. Check external service status
3. If service recovered, wait for recovery timeout
4. If persistent, check network connectivity and credentials

#### Performance Degradation
1. Check `/metrics` for latency percentiles
2. Check ML inference latency: `ml_inference_duration_seconds`
3. If LSTM is slow, consider disabling it or reducing sequence length
4. Scale horizontally: increase replica count
5. Check memory usage — LSTM and feature windows can be memory-intensive

### Alert Response

| Severity | Response Time | Action |
|----------|---------------|--------|
| `info` | Next business day | Log for trend analysis |
| `warning` | 4 hours | Investigate, check dashboards |
| `critical` | 15 minutes | Immediate investigation, consider auto-remediation |

### Backup & Recovery

- **PostgreSQL**: Daily pg_dump, 7-day retention
- **InfluxDB**: Daily backup, 30-day retention
- **Model files**: Versioned in models/trained/, backed up with PVC
- **Configuration**: Git versioned, environment secrets in Vault

### Scaling Guidelines

| Metric | Scale Up | Scale Down |
|--------|----------|------------|
| CPU > 70% | Add replicas | — |
| Memory > 80% | Add replicas or increase limits | — |
| Alert latency > 5s | Add replicas | — |
| < 10 alerts/hour | — | Reduce to min replicas |
| Inference p99 > 100ms | Check model complexity | — |
