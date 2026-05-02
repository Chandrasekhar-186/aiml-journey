# ML Model Monitor — Technical Architecture
Version: 1.0 | Phase 2 Final

## System Overview
Real-time ML model performance monitoring
using Databricks Lakehouse architecture.
Detects model drift in <1 minute.

## Data Flow

### Layer 1: Ingestion (Bronze)
Source: Kafka topic "ml-predictions"
Processing: Spark Structured Streaming
Output: Delta Lake Bronze table (append-only)
Guarantee: At-least-once (Kafka)
           Exactly-once (Delta sink)
Latency: <10 seconds

### Layer 2: Transformation (Silver)
Source: Bronze Delta table (batch)
Processing: PySpark DataFrame API
Quality checks:
  → Null score validation
  → Score range [0-100] enforcement
  → Latency outlier detection
Enrichment:
  → quality_tier (premium/standard/basic)
  → latency_bucket (fast/medium/slow)
  → is_correct (prediction accuracy flag)
Output: Delta Lake Silver (partitioned by model)
SLA: Processed within 5 minutes of Bronze write

### Layer 3: Aggregation (Gold)
Source: Silver Delta table
Processing: PySpark Window + GroupBy
Outputs:
  1. model_metrics: performance per model
  2. drift_detection: Z-score anomaly flags
Alert threshold: anomaly_rate > 5%
SLA: Gold updated every 5 minutes

### Layer 4: ML Tracking (MLflow)
Tracks: all Gold metrics per model per run
Tags: HEALTHY or DRIFT_DETECTED
Registry: best model auto-registered
Action: DRIFT_DETECTED → trigger retraining

### Layer 5: Graph Analysis (GraphFrames)
Model dependency graph
PageRank: identify critical models
Connected components: model clusters
Used for: impact analysis before model updates

## Key Design Decisions

### Why Delta Lake over Parquet?
ACID transactions prevent partial writes
during concurrent streaming + batch reads.
Time travel enables historical debugging.

### Why Z-score for drift detection?
Computationally O(n) rolling window.
Interpretable threshold (|z|>2.5 = anomaly).
No labeled drift data required.
Catches gradual AND sudden degradation.

### Why foreachBatch over standard sinks?
Enables writing to multiple sinks atomically.
Allows MLflow logging per batch.
Provides full DataFrame API for complex logic.

### Why Lakehouse over Lambda architecture?
Single pipeline for batch AND streaming.
No separate systems to maintain.
Delta handles both with identical APIs.
Simpler operations, fewer bugs.

## Performance Characteristics
Bronze ingestion:  20 events/second
Silver processing: ~1000 events/batch
Gold aggregation:  <30 seconds per run
MLflow logging:    <5 seconds per model
Total e2e latency: <2 minutes

## Tech Stack Summary
| Component | Technology | Version |
|-----------|-----------|---------|
| Compute | Apache Spark | 3.4+ |
| Storage | Delta Lake | 2.4+ |
| Streaming | Structured Streaming | 3.4+ |
| ML Tracking | MLflow | 2.x |
| Graph | GraphFrames | 0.8.2 |
| Orchestration | Databricks | DBR 13+ |
