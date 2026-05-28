# 🎯 Production CV Pipeline
> Real-time object detection at Databricks scale

## Problem
Process large-scale image datasets with
object detection, store results in Delta Lake,
track experiments with MLflow, serve via API.

## Architecture
Images (Delta Lake)
↓ Spark Pandas UDF
YOLOv8 Inference (distributed)
↓ Delta Lake (Bronze)
Result Parsing + Filtering
↓ Delta Lake (Silver)
Aggregated Statistics
↓ Delta Lake (Gold)
MLflow Experiment Tracking
↓ Model Registry
REST API Serving

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8n (Ultralytics) |
| Distributed | Apache Spark + Pandas UDF |
| Storage | Delta Lake (Bronze/Silver/Gold) |
| Tracking | MLflow |
| Embeddings | CLIP (optional) |

## Key Features
- Distributed YOLOv8 inference via Pandas UDF
- Lakehouse architecture (Bronze→Silver→Gold)
- Per-class statistics in Gold layer
- MLflow tracking + model registry
- Confidence threshold filtering
- Production patterns: idempotent writes

## Skills Demonstrated
✅ Computer Vision (YOLOv8)
✅ Apache Spark (distributed inference)
✅ Delta Lake (Lakehouse)
✅ MLflow (tracking + registry)
✅ Production patterns
✅ CV metrics (mAP, confidence)
