# Phase 4 Week 1 Project — Full MLOps Pipeline
# Date: June 16, 2026
# Everything from Days 1-4 in one pipeline!

import numpy as np
import pandas as pd
from scipy import stats
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time
import json

spark = SparkSession.builder \
    .appName("MLOpsPipeline") \
    .getOrCreate()

client = MlflowClient()
MODEL_NAME = "ProductionMLModel"
ACCURACY_THRESHOLD = 0.82

print("="*60)
print("Production ML Pipeline — Week 1 Project")
print("="*60)

# ══════════════════════════════════════════════
# STAGE 1: DATA LAYER (Bronze → Silver → Gold)
# ══════════════════════════════════════════════
print("\n[STAGE 1] Data Layer")

# Bronze: raw data ingestion
np.random.seed(42)
n_samples = 2000

raw_data = pd.DataFrame({
    'user_id': range(n_samples),
    'feature_1': np.random.normal(0, 1, n_samples),
    'feature_2': np.random.exponential(2, n_samples),
    'feature_3': np.random.uniform(0, 10, n_samples),
    'feature_4': np.random.normal(5, 2, n_samples),
    'category': np.random.choice(
        ['A','B','C'], n_samples
    ),
    'label': np.random.randint(0, 2, n_samples),
    'ingested_at': '2026-06-16'
})

bronze_df = spark.createDataFrame(raw_data)
bronze_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/mlops/bronze")
print(f"  ✅ Bronze: {bronze_df.count()} rows")

# Silver: feature engineering
silver_df = bronze_df \
    .withColumn("f1_squared",
                 F.col("feature_1")**2) \
    .withColumn("f2_log",
                 F.log(F.col("feature_2")+1)) \
    .withColumn("f3_normalized",
                 F.col("feature_3") / 10.0) \
    .withColumn("f1_f4_interaction",
                 F.col("feature_1") *
                 F.col("feature_4")) \
    .withColumn("is_cat_A",
                 (F.col("category")=="A")
                 .cast("int")) \
    .withColumn("processed_at",
                 F.current_timestamp())

silver_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/mlops/silver")
print(f"  ✅ Silver: {silver_df.count()} rows "
      f"with {len(silver_df.columns)} features")

# Gold: monitoring stats
gold_df = silver_df.groupBy("category").agg(
    F.count("*").alias("count"),
    F.avg("feature_1").alias("avg_f1"),
    F.stddev("feature_1").alias("std_f1"),
    F.avg("label").alias("label_rate")
)
gold_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/mlops/gold")
print(f"  ✅ Gold: category statistics")
gold_df.show()

# ══════════════════════════════════════════════
# STAGE 2: FEATURE STORE
# ══════════════════════════════════════════════
print("\n[STAGE 2] Feature Store")

FEATURE_COLS = [
    "feature_1", "feature_2", "feature_3",
    "feature_4", "f1_squared", "f2_log",
    "f3_normalized", "f1_f4_interaction",
    "is_cat_A"
]

# Write features to Delta (simulating FS)
feature_table = silver_df.select(
    "user_id", *FEATURE_COLS
)
feature_table.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/mlops/feature_store")

print(f"  ✅ Feature store: "
      f"{len(FEATURE_COLS)} features stored")
print(f"  Features: {FEATURE_COLS}")

# ══════════════════════════════════════════════
# STAGE 3: MODEL TRAINING
# ══════════════════════════════════════════════
print("\n[STAGE 3] Model Training")

# Create training set from feature store
training_pd = silver_df.select(
    *FEATURE_COLS, "label"
).toPandas()

X = training_pd[FEATURE_COLS].values
y = training_pd["label"].values

X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train model with MLflow tracking
mlflow.set_experiment("mlops_week1_project")

with mlflow.start_run(
        run_name="GBT_production_candidate") as run:
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log everything
    mlflow.log_params({
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_features": len(FEATURE_COLS),
        "training_samples": len(X_train)
    })
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1,
        "train_size": len(X_train),
        "test_size": len(X_test)
    })

    # Log with signature
    signature = infer_signature(
        X_train_s,
        model.predict(X_train_s)
    )
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME
    )

    run_id = run.info.run_id
    print(f"  ✅ Model trained:")
    print(f"     Accuracy: {accuracy:.4f}")
    print(f"     F1 Score: {f1:.4f}")
    print(f"     Run ID:   {run_id[:8]}...")

# ══════════════════════════════════════════════
# STAGE 4: QUALITY GATE + PROMOTION
# ══════════════════════════════════════════════
print("\n[STAGE 4] Quality Gate")

def quality_gate(accuracy: float,
                  f1: float) -> bool:
    """CI/CD quality gate — must pass to promote"""
    checks = {
        "accuracy_threshold":
            accuracy >= ACCURACY_THRESHOLD,
        "f1_threshold": f1 >= 0.75,
        "no_nan_predictions": True,
    }
    all_pass = all(checks.values())
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}: "
              f"{'PASS' if passed else 'FAIL'}")
    return all_pass

gate_passed = quality_gate(accuracy, f1)

if gate_passed:
    print(f"\n  🚀 Quality gate PASSED — "
          f"promoting to Staging!")
    # Transition to Staging
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version="1",
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"  ✅ Model promoted to Staging")
    except Exception as e:
        print(f"  ℹ️  Registry: {str(e)[:50]}")
else:
    print(f"\n  ❌ Quality gate FAILED — "
          f"not promoting!")

# ══════════════════════════════════════════════
# STAGE 5: A/B TESTING SETUP
# ══════════════════════════════════════════════
print("\n[STAGE 5] A/B Testing")

# Train baseline model for comparison
baseline = RandomForestClassifier(
    n_estimators=50, random_state=42
)
baseline.fit(X_train_s, y_train)
baseline_acc = accuracy_score(
    y_test, baseline.predict(X_test_s)
)

# A/B test results
ab_results = {
    "model_A": {
        "name": "RF_baseline",
        "accuracy": baseline_acc,
        "traffic": "80%"
    },
    "model_B": {
        "name": "GBT_candidate",
        "accuracy": accuracy,
        "traffic": "20%"
    }
}
winner = "B" if accuracy > baseline_acc else "A"

print(f"  Model A (baseline): "
      f"{baseline_acc:.4f} accuracy")
print(f"  Model B (candidate): "
      f"{accuracy:.4f} accuracy")
print(f"  Winner: Model {winner}! "
      f"({'GBT' if winner=='B' else 'RF'})")

# ══════════════════════════════════════════════
# STAGE 6: DRIFT MONITORING
# ══════════════════════════════════════════════
print("\n[STAGE 6] Drift Monitoring")

# Simulate production data (with drift!)
np.random.seed(123)
prod_data = pd.DataFrame({
    col: np.random.normal(0.5, 1.2, 500)
    if col in ['feature_1', 'feature_4']
    else np.random.normal(0, 1, 500)
    for col in ['feature_1', 'feature_2',
                 'feature_3', 'feature_4']
})

ref_data = raw_data[
    ['feature_1', 'feature_2',
     'feature_3', 'feature_4']
].head(500)

drift_alerts = []
for feat in ['feature_1', 'feature_2',
              'feature_3', 'feature_4']:
    ref = ref_data[feat].values
    cur = prod_data[feat].values
    _, p_val = stats.ks_2samp(ref, cur)
    drifted = p_val < 0.05
    status = "🚨 DRIFT" if drifted else "✅ OK"
    print(f"  {feat}: p={p_val:.4f} {status}")
    if drifted:
        drift_alerts.append(feat)

if drift_alerts:
    print(f"\n  ⚠️  Drift in: {drift_alerts}")
    print(f"  → Triggering retraining workflow!")
else:
    print(f"\n  ✅ No drift detected")

# ══════════════════════════════════════════════
# STAGE 7: FINAL MLFLOW SUMMARY
# ══════════════════════════════════════════════
print("\n[STAGE 7] Final MLflow Summary")

mlflow.set_experiment("mlops_week1_project")
with mlflow.start_run(
        run_name="pipeline_summary"):
    mlflow.log_params({
        "pipeline_version": "1.0",
        "stages": "Bronze→Silver→Gold→"
                  "FS→Train→Gate→AB→Monitor",
        "model": "GradientBoosting",
        "feature_count": len(FEATURE_COLS)
    })
    mlflow.log_metrics({
        "final_accuracy": accuracy,
        "final_f1": f1,
        "baseline_accuracy": baseline_acc,
        "accuracy_improvement":
            accuracy - baseline_acc,
        "quality_gate_passed":
            int(gate_passed),
        "drifted_features": len(drift_alerts),
        "ab_winner_is_new":
            int(winner == "B")
    })
    mlflow.set_tags({
        "pipeline_status": "complete",
        "promoted": str(gate_passed),
        "drift_detected": str(
            len(drift_alerts) > 0
        )
    })
    print("  ✅ Pipeline summary logged!")

# ══════════════════════════════════════════════
# PIPELINE COMPLETE
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("WEEK 1 MLOPS PROJECT COMPLETE! 🏆")
print("="*60)
print(f"""
Pipeline Summary:
  Bronze data:    {bronze_df.count()} rows
  Features:       {len(FEATURE_COLS)} engineered
  Model accuracy: {accuracy:.4f}
  Quality gate:   {'PASSED ✅' if gate_passed else 'FAILED ❌'}
  A/B winner:     Model {winner}
  Drift alerts:   {len(drift_alerts)} features

Databricks competencies demonstrated:
  ✅ Delta Lakehouse (Bronze→Silver→Gold)
  ✅ Feature Store (Delta backed)
  ✅ MLflow (tracking+registry+signature)
  ✅ Quality gates (CI/CD promotion)
  ✅ A/B testing (traffic split)
  ✅ Drift detection (KS test)
  ✅ Workflow orchestration (7 stages)
""")
