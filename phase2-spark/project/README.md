# Phase 2 Project — Real-time ML Model Monitor
Date: April 23, 2026
Timeline: Days 43-55 (incremental build)

## Project Overview
A production-grade system that monitors
ML model performance in real-time using
the complete Databricks stack.

## Architecture
Kafka (simulated events)
↓ Spark Structured Streaming
Bronze Delta Table (raw predictions)
↓ PySpark transformations
Silver Delta Table (enriched + validated)
↓ Windowed aggregations
Gold Delta Table (model metrics per minute)
↓ MLflow experiment comparison
Drift Detection Alert
↓ Delta CDF → Kafka alert topic

## Tech Stack
- Apache Spark (Structured Streaming)
- Delta Lake (Bronze/Silver/Gold)
- MLflow (experiment tracking + model registry)
- GraphFrames (model dependency graph)
- Databricks SQL (analytics queries)
- Kafka (event source + alert sink)

## Deliverables
Day 43: Project setup + Bronze layer
Day 46: Silver + Gold layers
Day 49: MLflow integration + drift detection
Day 52: GraphFrames model graph
Day 55: Full integration + README

## Why This Project Impresses Databricks
✅ Uses Lakehouse architecture
✅ Real-time streaming with Delta Lake
✅ MLflow for model monitoring
✅ Production patterns (idempotent, checkpoints)
✅ GraphFrames for model relationships
✅ Demonstrates platform-specific knowledge

# 🔍 Real-time ML Model Monitor
> Production-grade monitoring system for ML models
> using Databricks Lakehouse architecture

[![Spark](https://img.shields.io/badge/Apache_Spark-3.x-orange)]()
[![Delta](https://img.shields.io/badge/Delta_Lake-2.x-blue)]()
[![MLflow](https://img.shields.io/badge/MLflow-2.x-green)]()

## 🎯 Problem Statement
ML models degrade silently in production.
Without monitoring, accuracy drops go undetected
for days — causing business losses.

This system detects model drift in <1 minute
using real-time streaming on the Databricks
Lakehouse.

## 🏗️ Architecture
Kafka Events (predictions)
↓ Spark Structured Streaming
↓ Checkpoint: /tmp/checkpoint
Bronze Delta Table (raw, append-only)
↓ PySpark batch transform
↓ Quality validation
Silver Delta Table (clean, partitioned)
↓ Window aggregations
↓ Z-score drift detection
Gold Delta Table (metrics + alerts)
↓ MLflow experiment logging
Model Registry (best model versioned)
↓ Alert: DRIFT_DETECTED tag
Retraining trigger

## 📊 Key Results
| Model | Accuracy | Drift Rate | Status |
|-------|----------|------------|--------|
| RF    | 85.2%    | 2.0%       | ✅ Healthy |
| XGB   | 91.5%    | 8.0%       | 🚨 Alert |
| NN    | 88.7%    | 3.0%       | ✅ Healthy |

## 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| Streaming | Spark Structured Streaming |
| Storage | Delta Lake (Bronze/Silver/Gold) |
| ML Tracking | MLflow (metrics + registry) |
| Drift Detection | Z-score (rolling window) |
| Orchestration | Apache Spark |

## 🚀 Quick Start
```bash
# 1. Start Bronze ingestion
python bronze_layer.py

# 2. Process Silver + Gold
python silver_gold_layers.py

# 3. Log to MLflow
python mlflow_integration.py

# 4. View experiments
mlflow ui  # localhost:5000
```

## 💡 Key Technical Decisions
**Why Delta Lake over Parquet?**
ACID transactions prevent corrupt data during
concurrent streaming writes. Time travel enables
point-in-time debugging of model behavior.

**Why Z-score for drift detection?**
Computationally efficient (O(n) rolling window),
interpretable threshold (|z| > 2.5 = anomaly),
and works without labeled drift data.

**Why Lakehouse over Lambda?**
Single unified architecture. No separate batch
and streaming pipelines to maintain. Delta Lake
handles both with identical APIs.

## 📈 Skills Demonstrated
- ✅ Apache Spark (streaming + batch)
- ✅ Delta Lake (Lakehouse architecture)
- ✅ MLflow (tracking + model registry)
- ✅ Real-time drift detection
- ✅ Production patterns (checkpoints, idempotent)
- ✅ Data quality validation

---
*Built during 6-month Databricks prep journey*
*Day 45 of 180 — Phase 2 complete*


## 🏗️ Phase 2 Highlights (Apache Spark Deep Dive)

### Real-time ML Model Monitor
**Tech:** PySpark + Kafka + Delta Lake + MLflow + GraphFrames

**Architecture:** Bronze → Silver → Gold → MLflow → Alerts

**Key features:**
- Detects model drift in <1 minute (Z-score)
- Exactly-once streaming with Delta ACID
- 7 Databricks competencies in one project
- Production patterns: checkpoints, idempotent writes

**[View Project →](phase2-spark/project/)**

### Phase 2 Certifications Earned
- ✅ Databricks Generative AI Fundamentals
- ✅ Databricks MLflow Fundamentals

### Phase 2 Stats
- 30 days of Spark deep dive
- 40+ LeetCode problems (Phase 2)
- 4 full cert mock exams (scoring 83-88%)
- 1 production-grade streaming project
