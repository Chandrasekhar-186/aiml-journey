# 🏭 Production ML Pipeline
> End-to-end MLOps: Training → Serving → Monitoring

## Problem
Build a production-grade ML pipeline that
trains, serves, monitors, and automatically
retrains a classification model on Databricks.

## Architecture
Raw Data (Delta Lake Bronze)
↓ Feature Engineering (Silver)
Feature Store (Delta Lake)
↓ Training Set Creation
Model Training (MLflow tracked)
↓ Quality Gate (accuracy > 0.85)
MLflow Model Registry (Staging → Production)
↓ REST API Serving
Live Predictions
↓ Drift Detection (KS + PSI)
Monitoring Dashboard (Gold layer)
↓ Alert if drift detected
Automated Retraining Trigger (Workflows)

## Components
| Day | Component | Status |
|-----|-----------|--------|
| 1 | MLflow serving + A/B | ✅ |
| 2 | Feature store + monitoring | ✅ |
| 3 | Databricks Workflows + DLT | ✅ |
| 4 | CI/CD + testing | ✅ |
| 5 | Full integration project | ✅ today |

## Skills Demonstrated
✅ MLflow (tracking + registry + serving)
✅ Feature Store (Delta Lake backed)
✅ Drift detection (KS + PSI)
✅ Databricks Workflows (DAG orchestration)
✅ CI/CD (pytest + promotion gates)
✅ Delta Lakehouse (Bronze/Silver/Gold)
✅ A/B testing for ML models
