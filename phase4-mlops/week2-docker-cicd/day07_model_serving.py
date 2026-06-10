# Phase 4 Day 7 — Databricks Model Serving
# Date: June 18, 2026
# Production inference at Databricks scale!

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time
import json

spark = SparkSession.builder \
    .appName("ModelServing") \
    .getOrCreate()

print("="*60)
print("Databricks Model Serving + Inference")
print("="*60)

"""
DATABRICKS MODEL SERVING

What it is:
→ Managed REST endpoint for MLflow models
→ Auto-scales based on traffic
→ Zero infrastructure management
→ Built-in A/B testing
→ Latency + throughput metrics
→ Integrated with Unity Catalog

How it works:
1. Register model in MLflow Model Registry
2. Create serving endpoint in UI or API
3. Configure: model version + cluster size
4. Databricks handles: scaling, health, SSL
5. Call endpoint: POST /serving-endpoints/{name}/invocations

Endpoint sizes:
Small:  1-4 CPUs, 4-8GB RAM  (dev/test)
Medium: 4-8 CPUs, 8-16GB RAM (production)
Large:  8-16 CPUs, 16-32GB RAM (heavy load)
GPU:    T4/A10G for deep learning!

Scale to zero:
→ No traffic = 0 cost!
→ Cold start: 30-60 seconds
→ Disable for latency-critical workloads

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONLINE vs BATCH INFERENCE

Online (real-time) inference:
→ Low latency (<100ms typical)
→ One prediction at a time
→ Triggered by user action
→ Databricks Model Serving
→ FastAPI + Docker
→ Use: recommendations, fraud detection

Batch inference:
→ High throughput (millions of records)
→ Scheduled runs (nightly, weekly)
→ Spark Pandas UDF for distribution
→ Results stored in Delta Lake
→ Use: churn prediction, credit scoring

Streaming inference:
→ Continuous processing
→ Spark Structured Streaming + model
→ Predictions as stream arrives
→ Use: real-time fraud, monitoring

Decision guide:
Need result in <1 second?  → Online
Processing historical data? → Batch
Continuous data stream?     → Streaming

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REAL-TIME FEATURE PIPELINES

Challenge: online inference needs features
           computed in <100ms!

Solutions:

1. Pre-computed features (offline store):
   → Run batch job nightly
   → Store features in Redis/Delta
   → Lookup at inference time: O(1)!
   → Works when features don't change fast

2. On-demand features (online compute):
   → Compute features at request time
   → Must be FAST (simple operations only)
   → Use: age of account, current session

3. Streaming features (near real-time):
   → Compute features from stream
   → 1-5 minute latency
   → Kafka → Spark Streaming → Feature Store
   → Use: recent activity counts

Databricks Online Tables:
→ Sync Delta table → low-latency store
→ Single-digit ms lookups!
→ Perfect for feature serving
"""

# 1. Train + register model
print("\n=== TRAIN + REGISTER MODEL ===")

X, y = make_classification(
    n_samples=3000, n_features=10,
    n_informative=7, random_state=42
)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.2,
                     random_state=42)

mlflow.set_experiment("phase4_model_serving")

with mlflow.start_run(
        run_name="serving_demo_model"):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(
        y_test, model.predict(X_test)
    )

    from mlflow.models import infer_signature
    sig = infer_signature(
        X_train, model.predict(X_train)
    )

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=sig,
        registered_model_name="ServingDemo"
    )
    mlflow.log_metric("accuracy", acc)
    print(f"  Model registered: acc={acc:.4f}")

# 2. Databricks serving endpoint config
print("\n=== DATABRICKS ENDPOINT CONFIG ===")
endpoint_config = {
    "name": "ml-production-endpoint",
    "config": {
        "served_models": [
            {
                "model_name": "ServingDemo",
                "model_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
                "environment_vars": {
                    "ENABLE_MLFLOW_TRACING": "true"
                }
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name":
                        "ServingDemo-1",
                    "traffic_percentage": 100
                }
            ]
        }
    }
}
print(json.dumps(endpoint_config, indent=2))

# 3. Calling the endpoint
print("\n=== CALLING SERVING ENDPOINT ===")
print("""
import requests

# Get Databricks token
token = dbutils.notebook.entry_point \\
    .getDbutils().notebook().getContext() \\
    .apiToken().get()

# Call endpoint
url = "https://<workspace>.azuredatabricks.net"
    "/serving-endpoints/ml-production-endpoint"
    "/invocations"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Single prediction
data = {
    "inputs": [[1.2, 0.5, 3.1, 2.0, 1.4,
                0.4, 0.3, 2.4, 1.0, 0.8]]
}
response = requests.post(url,
    headers=headers,
    json=data
)
prediction = response.json()["predictions"]
print(f"Prediction: {prediction}")

# Batch predictions (up to 256 rows)
data_batch = {
    "inputs": X_test[:10].tolist()
}
response = requests.post(url,
    headers=headers,
    json=data_batch
)
predictions = response.json()["predictions"]
print(f"Batch predictions: {predictions}")
""")

# 4. Batch inference on Spark
print("\n=== BATCH INFERENCE ON SPARK ===")

# Load test data as Spark DataFrame
test_pdf = pd.DataFrame(
    X_test,
    columns=[f"f{i}" for i in range(10)]
)
test_df = spark.createDataFrame(test_pdf)

# Broadcast model (send to all executors)
import mlflow.sklearn
from pyspark.sql.types import IntegerType

# Load and broadcast model
loaded_model = mlflow.sklearn.load_model(
    f"runs:/{mlflow.last_active_run().info.run_id}"
    f"/model"
)

# Create predict UDF
feature_cols = [f"f{i}" for i in range(10)]

predict_udf = mlflow.pyfunc\
    .spark_udf(spark, f"runs:/"
               f"{mlflow.last_active_run().info.run_id}"
               f"/model")

# Apply to Spark DataFrame
predictions_df = test_df.withColumn(
    "prediction",
    predict_udf(*[F.col(c)
                   for c in feature_cols])
)

print("Batch inference on Spark:")
predictions_df.show(5)

# Write predictions to Delta
predictions_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/batch_predictions")

pred_count = predictions_df.count()
print(f"  ✅ {pred_count} predictions"
      f" written to Delta Lake")

# 5. Streaming inference
print("\n=== STREAMING INFERENCE ===")
print("""
# Real-time inference on streaming data!
# Requires streaming source (Kafka/Delta)

from pyspark.sql import functions as F

# Read stream
stream_df = (spark.readStream
    .format("delta")
    .load("/incoming/features/"))

# Apply model (same UDF as batch!)
predictions_stream = stream_df.withColumn(
    "prediction",
    predict_udf(*feature_cols)
).withColumn(
    "predicted_at",
    F.current_timestamp()
)

# Write predictions stream
query = (predictions_stream.writeStream
    .format("delta")
    .option("checkpointLocation",
            "/checkpoints/predictions/")
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start("/streaming/predictions/"))

# Same model, same UDF
# Works for batch AND streaming!
# This is Spark's unification power!
""")

# 6. Online table for feature serving
print("\n=== ONLINE TABLES FOR FEATURES ===")
print("""
# Databricks Online Tables
# Sync Delta → low-latency key-value store
# Perfect for feature serving!

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec
)

w = WorkspaceClient()

# Create online table from Delta
online_table = w.online_tables.create(
    name="main.features.user_features_online",
    spec=OnlineTableSpec(
        source_table_full_name=
            "main.features.user_features",
        primary_key_columns=["user_id"],
        # Sync every 5 minutes
        run_triggered={}
    )
)

# Lookup features at inference time
# Single-digit millisecond latency!
feature = w.online_tables.get_data(
    table_name=
        "main.features.user_features_online",
    key={"user_id": "user_123"}
)
print(f"Features: {feature}")
""")

# 7. Latency benchmark
print("\n=== INFERENCE LATENCY BENCHMARK ===")
import time

n_requests = 100
latencies = []

for i in range(n_requests):
    x = np.random.randn(1, 10)
    start = time.perf_counter()
    pred = model.predict(x)
    elapsed = (time.perf_counter() - start) * 1000
    latencies.append(elapsed)

latencies = np.array(latencies)
print(f"Latency over {n_requests} requests:")
print(f"  p50: {np.percentile(latencies,50):.3f}ms")
print(f"  p95: {np.percentile(latencies,95):.3f}ms")
print(f"  p99: {np.percentile(latencies,99):.3f}ms")
print(f"  max: {latencies.max():.3f}ms")

mlflow.set_experiment("phase4_model_serving")
with mlflow.start_run(
        run_name="serving_benchmarks"):
    mlflow.log_metrics({
        "p50_latency_ms":
            float(np.percentile(latencies, 50)),
        "p95_latency_ms":
            float(np.percentile(latencies, 95)),
        "p99_latency_ms":
            float(np.percentile(latencies, 99)),
        "throughput_rps":
            1000 / float(np.mean(latencies))
    })
    print("\nLatency benchmarks logged!")

print("\n" + "="*60)
print("Databricks Model Serving — MASTERED! 🚀")
print("Phase 4 Day 7 COMPLETE!")
print("="*60)
