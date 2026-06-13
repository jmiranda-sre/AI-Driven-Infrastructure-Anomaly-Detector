# Contributing to AI-Driven Infrastructure Anomaly Detector

Thank you for your interest in contributing! This document provides guidelines and instructions.

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## How to Contribute

### Bug Reports
1. Check if the issue already exists in [Issues](../../issues)
2. Open a new issue with: **bug** label, steps to reproduce, expected vs actual behavior, environment details

### Feature Requests
1. Open an issue with the **enhancement** label
2. Describe the use case, expected behavior, and why it adds value
3. Wait for maintainer feedback before implementing

### Pull Requests
1. Fork the repository
2. Create a feature branch: `feat/your-feature-name`
3. Write code with tests
4. Ensure all CI checks pass
5. Submit PR against `develop` branch

## Development Setup

```bash
# Clone and setup
git clone https://github.com/your-org/AI-Driven_Infrastructure_Anomaly_Detector.git
cd AI-Driven_Infrastructure_Anomaly_Detector
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,full]"

# Run tests
pytest tests/unit -v
pytest tests/integration -v

# Run linters
ruff check src/ tests/
ruff format src/ tests/
mypy src/

# Security scan
pip-audit
bandit -r src/ -ll
```

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(ml): add DBSCAN clustering model
fix(alerting): prevent alert storm on server group
docs(api): update endpoint documentation
test(ingestion): add kafka consumer integration test
refactor(core): extract circuit breaker to standalone module
perf(ml): optimize isolation forest batch inference
```

## Architecture Overview

| Module | Purpose |
|--------|---------|
| `src/core/` | Errors, config, logging, security, health, DB, circuit breaker |
| `src/ingestion/` | Prometheus client, Kafka consumer, feature engineering, orchestrator |
| `src/ml_processing/` | Models (IF, LSTM, HW), ensemble, drift detection, pipeline, model manager |
| `src/alerting/` | Alert evaluation, dispatch (Slack, PD, webhook, email), service |
| `src/api/` | FastAPI routes, schemas, error handlers, app factory |

## Adding a New ML Model

1. Create `src/ml_processing/your_model.py` implementing the `AnomalyModel` protocol
2. Register in `MLPipeline._init_models()`
3. Add weight in ensemble config
4. Add training script support in `models/scripts/train.py`
5. Write unit tests in `tests/unit/test_ml_processing.py`
6. Update `config/default.yaml` with model config section

## Adding a New Alert Channel

1. Create a class implementing `AlertChannel` in `src/alerting/dispatcher.py`
2. Add config in `config/default.yaml` under `alerting.channels`
3. Register in `AlertDispatcher.__init__()`
4. Write tests

## Governance Model

- **Maintainers**: Have merge access. Review PRs and guide architecture decisions.
- **Contributors**: Submit PRs, report issues, participate in discussions.
- **Decisions**: Made by maintainers based on community feedback. Major changes require an RFC issue.

## Release Process

1. Update version in `pyproject.toml` and `config/default.yaml`
2. Generate changelog from conventional commits
3. Create GitHub release with tag
4. CI/CD builds and pushes Docker image automatically
