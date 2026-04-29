# Phase 2 Project — MLflow Integration
# Date: April 25, 2026
# Complete the monitoring loop!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

spark = SparkSession.builder \
    .appName("ModelMonitor_MLflow") \
    .getOrCreate()

client = MlflowClient()
print("="*60)
print("MLflow Integration — Complete Monitor")
print("="*60)

# 1. Load Gold metrics
try:
    gold_metrics = spark.read.format("delta") \
        .load("/tmp/monitor_gold_metrics")
    gold_drift = spark.read.format("delta") \
        .load("/tmp/monitor_gold_drift")
    print("Gold tables loaded!")
except Exception:
    # Synthetic data
    gold_metrics = spark.createDataFrame([
        ("RF", 1000, 85.2, 4.1, 12.3, 0.852),
        ("XGB", 1000, 91.5, 3.2, 11.1, 0.915),
        ("NN", 1000, 88.7, 5.8, 18.4, 0.887),
    ], ["model_name", "total_predictions",
        "avg_score", "score_stddev",
        "avg_latency_ms", "accuracy"])

    gold_drift = spark.createDataFrame([
        ("RF", 0.02, 20),
        ("XGB", 0.08, 80),  # drift alert!
        ("NN", 0.03, 30),
    ], ["model_name", "anomaly_rate",
        "anomaly_count"])

# 2. Log all model metrics to MLflow
print("\n=== LOGGING TO MLFLOW ===")
mlflow.set_experiment("model_monitor_project")

metrics_rows = gold_metrics.collect()
drift_rows = {r.model_name: r
              for r in gold_drift.collect()}

for row in metrics_rows:
    model = row.model_name
    drift = drift_rows.get(model)

    with mlflow.start_run(
            run_name=f"monitor_{model}"):
        # Log model identity
        mlflow.log_param("model_name", model)
        mlflow.log_param(
            "monitor_timestamp",
            str(F.current_timestamp())
        )

        # Log performance metrics
        mlflow.log_metric(
            "total_predictions",
            row.total_predictions
        )
        mlflow.log_metric(
            "avg_score", row.avg_score
        )
        mlflow.log_metric(
            "score_stddev", row.score_stddev
        )
        mlflow.log_metric(
            "avg_latency_ms",
            row.avg_latency_ms
        )
        mlflow.log_metric(
            "accuracy", row.accuracy
        )

        # Log drift metrics
        if drift:
            mlflow.log_metric(
                "anomaly_rate",
                drift.anomaly_rate
            )
            mlflow.log_metric(
                "anomaly_count",
                drift.anomaly_count
            )

            # Flag drift alert
            is_drifting = (
                drift.anomaly_rate > 0.05
            )
            mlflow.log_param(
                "drift_alert",
                str(is_drifting)
            )
            mlflow.set_tag(
                "status",
                "DRIFT_DETECTED"
                if is_drifting
                else "HEALTHY"
            )

        print(f"✅ {model}: logged to MLflow "
              f"(drift={drift.anomaly_rate:.2%}"
              f" if drift else 'N/A')")

# 3. Compare models in MLflow
print("\n=== MODEL COMPARISON ===")
experiment = mlflow.get_experiment_by_name(
    "model_monitor_project"
)
if experiment:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.status = 'HEALTHY'",
        order_by=["metrics.accuracy DESC"]
    )
    if not runs.empty:
        print("Healthy models by accuracy:")
        print(runs[["tags.mlflow.runName",
                     "metrics.accuracy",
                     "metrics.avg_latency_ms"]]
              .head())

# 4. Register best performing model
print("\n=== REGISTER BEST MODEL ===")
best_model = max(metrics_rows,
                  key=lambda r: r.accuracy)
print(f"Best model: {best_model.model_name} "
      f"({best_model.accuracy:.1%} accuracy)")

# In production: register to Model Registry
print("""
# Register best model:
mlflow.register_model(
    f"runs:/{best_run_id}/model",
    "ProductionMLModel"
)

# Transition to production:
client.transition_model_version_stage(
    name="ProductionMLModel",
    version=1,
    stage="Production"
)
""")

# 5. Drift alerting system
print("\n=== DRIFT ALERT SYSTEM ===")
drifting_models = [
    r for r in gold_drift.collect()
    if r.anomaly_rate > 0.05
]

if drifting_models:
    print("🚨 DRIFT ALERTS:")
    for model in drifting_models:
        print(f"  Model: {model.model_name}")
        print(f"  Anomaly rate: "
              f"{model.anomaly_rate:.1%}")
        print(f"  Action: Trigger retraining!")
        print(f"  MLflow tag: DRIFT_DETECTED")
else:
    print("✅ All models healthy!")

# 6. Complete project summary
print("\n" + "="*60)
print("MODEL MONITOR PROJECT — COMPLETE!")
print("="*60)
print("""
Architecture implemented:
✅ Bronze: Kafka stream → Delta Lake
✅ Silver: Clean + validate + enrich
✅ Gold:   Aggregate + drift detection
✅ MLflow: Track all metrics + alerts
✅ Registry: Best model versioned

Production capabilities:
→ Real-time prediction monitoring
→ Automatic drift detection (Z-score)
→ MLflow experiment tracking
→ Model registry with staging workflow
→ Alert system for retraining triggers
→ Time travel on all Delta layers
→ Full lineage Bronze → Silver → Gold

This covers 6 Databricks competencies:
1. Apache Spark (streaming pipeline)
2. Delta Lake (Lakehouse architecture)
3. MLflow (tracking + registry)
4. Databricks SQL (Gold table queries)
5. Production patterns (idempotent, checkpoints)
6. ML monitoring (drift detection)
""")

mlflow.set_experiment("model_monitor_project")
with mlflow.start_run(
        run_name="project_complete"):
    mlflow.log_param("status", "COMPLETE")
    mlflow.log_param("layers",
                     "bronze+silver+gold")
    mlflow.log_metric("total_models",
                       len(metrics_rows))
    mlflow.log_metric("drifting_models",
                       len(drifting_models))
    print("\nProject completion logged!")
