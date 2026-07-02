# 🚀 Production ML Serving System
> Containerized model serving with full observability

## Problem
Deploy a trained ML model as a production
REST API with monitoring, health checks,
CI/CD, and multiple inference patterns.

## Architecture
MLflow Model Registry (Production stage)

↓ Load at startup

FastAPI REST API (Docker container)

├── /health (liveness probe)

├── /predict (online inference)

├── /predict/batch (batch REST)

└── /metrics (Prometheus)

↓

Prometheus (metrics collection)

↓

Grafana (monitoring dashboard)
Databricks Model Serving (managed)

↓ Same model, managed infrastructure

Spark Batch Inference (scheduled)

↓ Delta Lake predictions table

## Skills Demonstrated
✅ Docker (Dockerfile + Compose)
✅ FastAPI (async, Pydantic, startup)
✅ Prometheus metrics (counter, histogram)
✅ Databricks Model Serving (config)
✅ Online + Batch inference patterns
✅ p50/p95/p99 latency benchmarking
✅ MLflow integration (load + serve)
✅ Production health checks
