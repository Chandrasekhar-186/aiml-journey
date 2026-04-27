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
