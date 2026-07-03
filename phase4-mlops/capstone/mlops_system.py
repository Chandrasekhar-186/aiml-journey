# Phase 4 Capstone — Complete MLOps System
# Date: June 21, 2026
# ALL Phase 4 concepts in one system!

import numpy as np
import pandas as pd
from scipy import stats
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import logging
import json
import time
import os
from datetime import datetime

spark = SparkSession.builder \
    .appName("MLOpsCapstone") \
    .getOrCreate()

client = MlflowClient()
MODEL_NAME = "CapstoneMLModel"
ACCURACY_GATE = 0.83
F1_GATE = 0.80

print("="*60)
print("Phase 4 Capstone — Complete MLOps System")
print("10 Databricks Competencies")
print("="*60)

# ══════════════════════════════════════════
# COMPETENCY 1: DELTA LAKEHOUSE
# ══════════════════════════════════════════
print("\n[C1] Delta Lakehouse Architecture")

np.random.seed(42)
N = 3000
raw = pd.DataFrame({
    'id': range(N),
    'f1': np.random.normal(0, 1, N),
    'f2': np.random.exponential(2, N),
    'f3': np.random.uniform(0, 10, N),
    'f4': np.random.normal(5, 2, N),
    'f5': np.random.beta(2, 5, N),
    'cat': np.random.choice(['A','B','C'], N),
    'label': np.random.randint(0, 2, N),
    'date': '2026-06-21'
})

# Bronze
bronze = spark.createDataFrame(raw)
bronze.write.format("delta").mode("overwrite") \
    .save("/tmp/capstone/bronze")

# Silver: engineered features
silver = bronze \
    .withColumn("f1_sq", F.col("f1")**2) \
    .withColumn("f2_log", F.log(F.col("f2")+1)) \
    .withColumn("f3_norm", F.col("f3")/10) \
    .withColumn("f1_f4", F.col("f1")*F.col("f4")) \
    .withColumn("is_A", (F.col("cat")=="A")
                .cast("int")) \
    .withColumn("processed_at",
                 F.current_timestamp())
silver.write.format("delta").mode("overwrite") \
    .partitionBy("date") \
    .save("/tmp/capstone/silver")

# Gold
gold = silver.groupBy("cat").agg(
    F.count("*").alias("n"),
    F.avg("f1").alias("avg_f1"),
    F.avg("label").alias("label_rate")
)
gold.write.format("delta").mode("overwrite") \
    .save("/tmp/capstone/gold")

print(f"  ✅ Bronze: {bronze.count()} rows")
print(f"  ✅ Silver: {silver.count()} rows, "
      f"{len(silver.columns)} cols")
print(f"  ✅ Gold: {gold.count()} categories")

# ══════════════════════════════════════════
# COMPETENCY 2: FEATURE STORE
# ══════════════════════════════════════════
print("\n[C2] Feature Store")

FEATURES = [
    'f1','f2','f3','f4','f5',
    'f1_sq','f2_log','f3_norm','f1_f4','is_A'
]

feature_table = silver.select('id', *FEATURES)
feature_table.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/capstone/feature_store")

print(f"  ✅ {len(FEATURES)} features stored")
print(f"  Features: {FEATURES}")

# ══════════════════════════════════════════
# COMPETENCY 3: MLFLOW TRACKING
# ══════════════════════════════════════════
print("\n[C3] MLflow Tracking + Registry")

train_pd = silver.select(
    *FEATURES, 'label'
).toPandas()

X = train_pd[FEATURES].values
y = train_pd['label'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

mlflow.set_experiment("phase4_capstone")

with mlflow.start_run(
        run_name="capstone_model") as run:
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=42
    )
    model.fit(X_tr_s, y_tr)
    acc = accuracy_score(
        y_te, model.predict(X_te_s)
    )
    f1 = f1_score(y_te, model.predict(X_te_s))

    sig = infer_signature(
        X_tr_s, model.predict(X_tr_s)
    )
    mlflow.log_params({
        "n_estimators": 100,
        "n_features": len(FEATURES)
    })
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_score": f1
    })
    mlflow.sklearn.log_model(
        model, "model",
        signature=sig,
        registered_model_name=MODEL_NAME
    )
    RUN_ID = run.info.run_id
    print(f"  ✅ acc={acc:.4f} f1={f1:.4f}")

# ══════════════════════════════════════════
# COMPETENCY 4: A/B TESTING
# ══════════════════════════════════════════
print("\n[C4] A/B Testing Framework")

from sklearn.ensemble import RandomForestClassifier

baseline = RandomForestClassifier(
    n_estimators=50, random_state=42
)
baseline.fit(X_tr_s, y_tr)
base_acc = accuracy_score(
    y_te, baseline.predict(X_te_s)
)

np.random.seed(99)
n_requests = 500
ab_log = []
for _ in range(n_requests):
    model_choice = (
        "GBT" if np.random.random() < 0.8
        else "RF"
    )
    x = X_te_s[
        np.random.randint(len(X_te_s))
    ].reshape(1, -1)
    pred = (
        model.predict(x)[0] if model_choice == "GBT"
        else baseline.predict(x)[0]
    )
    ab_log.append({
        "model": model_choice,
        "prediction": pred
    })

ab_df = pd.DataFrame(ab_log)
print(f"  GBT accuracy:  {acc:.4f} (20% traffic)")
print(f"  RF accuracy:   {base_acc:.4f} (80% traffic)")
winner = "GBT" if acc > base_acc else "RF"
print(f"  A/B Winner:    {winner}")

# ══════════════════════════════════════════
# COMPETENCY 5: DRIFT DETECTION
# ══════════════════════════════════════════
print("\n[C5] Drift Detection (KS + PSI)")

np.random.seed(123)
prod_data = pd.DataFrame({
    'f1': np.random.normal(0.8, 1.2, 500),
    'f2': np.random.exponential(2.5, 500),
    'f3': np.random.uniform(1, 11, 500),
})
ref_data = raw[['f1','f2','f3']].head(500)

drift_results = {}
for feat in ['f1','f2','f3']:
    _, p = stats.ks_2samp(
        ref_data[feat], prod_data[feat]
    )
    drifted = p < 0.05
    drift_results[feat] = {
        "p_value": p, "drifted": drifted
    }
    status = "🚨" if drifted else "✅"
    print(f"  {feat}: p={p:.4f} {status}")

drifted_count = sum(
    v["drifted"] for v in drift_results.values()
)
retrain_needed = drifted_count >= 2
print(f"  Retrain needed: {retrain_needed}")

# ══════════════════════════════════════════
# COMPETENCY 6: FASTAPI + DOCKER
# ══════════════════════════════════════════
print("\n[C6] FastAPI + Docker Config")
print("""
  Dockerfile: python:3.10-slim
  API:        /health /predict /metrics
  Middleware: request tracking + latency
  Metrics:    Counter + Histogram + Gauge
  Deploy:     docker-compose up -d
  Scale:      --scale ml-api=3
  ✅ Full production FastAPI app written (Day 6)
""")

# ══════════════════════════════════════════
# COMPETENCY 7: SPARK BATCH INFERENCE
# ══════════════════════════════════════════
print("\n[C7] Spark Batch Inference")

test_pdf = pd.DataFrame(
    X_te_s,
    columns=[f"f{i}" for i in range(len(FEATURES))]
)
test_sdf = spark.createDataFrame(test_pdf)

predict_udf = mlflow.pyfunc.spark_udf(
    spark, f"runs:/{RUN_ID}/model",
    result_type="integer"
)
batch_df = test_sdf.withColumn(
    "prediction",
    predict_udf(*[F.col(c)
                   for c in test_sdf.columns])
).withColumn(
    "batch_ts", F.current_timestamp()
)
batch_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/capstone/predictions")

print(f"  ✅ {batch_df.count()} batch predictions")

# ══════════════════════════════════════════
# COMPETENCY 8: DATABRICKS WORKFLOWS
# ══════════════════════════════════════════
print("\n[C8] Databricks Workflows DAG")

workflow = {
    "name": "MLOps_Capstone_Pipeline",
    "schedule": "0 2 * * *",
    "tasks": [
        {"key": "ingest", "type": "notebook",
         "path": "/Workflows/01_ingest"},
        {"key": "features",
         "depends_on": ["ingest"],
         "type": "notebook",
         "path": "/Workflows/02_features"},
        {"key": "train",
         "depends_on": ["features"],
         "type": "python_wheel",
         "entry": "train"},
        {"key": "quality_gate",
         "depends_on": ["train"],
         "type": "condition",
         "expr": "accuracy > 0.83"},
        {"key": "deploy",
         "depends_on": ["quality_gate"],
         "type": "notebook",
         "path": "/Workflows/04_deploy"}
    ]
}
print(f"  ✅ {len(workflow['tasks'])}-task "
      f"DAG configured")
for t in workflow['tasks']:
    deps = t.get('depends_on', ['start'])
    print(f"     {' → '.join(deps)} → {t['key']}")

# ══════════════════════════════════════════
# COMPETENCY 9: OBSERVABILITY
# ══════════════════════════════════════════
print("\n[C9] Structured Observability")

latencies = []
for _ in range(200):
    x = X_te_s[
        np.random.randint(len(X_te_s))
    ].reshape(1, -1)
    t0 = time.perf_counter()
    model.predict(x)
    latencies.append(
        (time.perf_counter()-t0)*1000
    )

latencies = np.array(latencies)
error_rate = 0.003  # simulated

print(f"  p50:        {np.percentile(latencies,50):.2f}ms")
print(f"  p99:        {np.percentile(latencies,99):.2f}ms")
print(f"  Error rate: {error_rate:.1%}")
print(f"  SLA p99<200ms: "
      f"{'✅ PASS' if np.percentile(latencies,99)<200 else '❌ FAIL'}")

# ══════════════════════════════════════════
# COMPETENCY 10: CI/CD GATE
# ══════════════════════════════════════════
print("\n[C10] CI/CD Automated Promotion")

gates = {
    "accuracy_gate": acc >= ACCURACY_GATE,
    "f1_gate": f1 >= F1_GATE,
    "latency_gate":
        np.percentile(latencies, 99) < 200,
    "error_rate_gate": error_rate < 0.01,
    "drift_gate": drifted_count < 2
}

all_pass = all(gates.values())
for gate, passed in gates.items():
    print(f"  {'✅' if passed else '❌'} {gate}")

print(f"\n  Decision: "
      f"{'🚀 PROMOTE TO PRODUCTION' if all_pass else '❌ BLOCK DEPLOYMENT'}")

# ══════════════════════════════════════════
# FINAL MLFLOW SUMMARY
# ══════════════════════════════════════════
print("\n[FINAL] MLflow Capstone Summary")

with mlflow.start_run(
        run_name="capstone_summary"):
    mlflow.log_params({
        "competencies": 10,
        "pipeline_stages": 7,
        "n_features": len(FEATURES),
        "model": "GradientBoosting",
        "workflow_tasks": 5
    })
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_score": f1,
        "baseline_accuracy": base_acc,
        "drifted_features": drifted_count,
        "p99_latency_ms": float(
            np.percentile(latencies, 99)
        ),
        "all_gates_passed": int(all_pass),
        "ab_winner_is_new": int(
            winner == "GBT"
        )
    })
    mlflow.set_tags({
        "phase": "4_capstone",
        "promoted": str(all_pass),
        "competencies_demonstrated": "10"
    })
    print("  ✅ Capstone summary logged!")

# ══════════════════════════════════════════
# COMPLETION SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 4 CAPSTONE COMPLETE! 🏆")
print("="*60)
print(f"""
10 Databricks Competencies Demonstrated:
  C1.  ✅ Delta Lakehouse (B→S→G)
  C2.  ✅ Feature Store
  C3.  ✅ MLflow tracking + registry
  C4.  ✅ A/B testing
  C5.  ✅ Drift detection
  C6.  ✅ FastAPI + Docker
  C7.  ✅ Spark batch inference
  C8.  ✅ Databricks Workflows
  C9.  ✅ Structured observability
  C10. ✅ CI/CD promotion gates

Model: acc={acc:.4f} f1={f1:.4f}
Gate:  {'PASS → PROMOTED' if all_pass else 'FAIL → BLOCKED'}
""")
