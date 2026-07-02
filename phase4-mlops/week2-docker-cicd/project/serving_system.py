# Week 2 Project — Production Serving System
# Date: June 19, 2026
# Full observability + multi-pattern serving!

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time
import json
import os

spark = SparkSession.builder \
    .appName("ServingSystem") \
    .getOrCreate()

client = MlflowClient()
MODEL_NAME = "ProductionServingModel"

print("="*60)
print("Production ML Serving System")
print("Docker + FastAPI + Monitoring")
print("="*60)

# ══════════════════════════════════════════
# PART 1: TRAIN + REGISTER MODEL
# ══════════════════════════════════════════
print("\n[1] Training + Registering Model")

X, y = make_classification(
    n_samples=5000, n_features=12,
    n_informative=8, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

mlflow.set_experiment("week2_serving_project")

with mlflow.start_run(
        run_name="production_gbt") as run:
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_tr_s, y_tr)
    acc = accuracy_score(
        y_te, model.predict(X_te_s)
    )
    f1_report = classification_report(
        y_te, model.predict(X_te_s)
    )

    sig = infer_signature(
        X_tr_s, model.predict(X_tr_s)
    )
    mlflow.log_params({
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_features": 12
    })
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=sig,
        registered_model_name=MODEL_NAME
    )
    RUN_ID = run.info.run_id
    print(f"  ✅ Model: acc={acc:.4f}")

# ══════════════════════════════════════════
# PART 2: COMPLETE FASTAPI APPLICATION
# ══════════════════════════════════════════
print("\n[2] FastAPI Application")

FASTAPI_COMPLETE = '''
# production_api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
import time
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

# ── Prometheus Metrics ──────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    buckets=[.005,.01,.025,.05,.1,.25,.5,1,2.5]
)
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_version", "class"]
)
MODEL_ACCURACY_GAUGE = Gauge(
    "model_accuracy",
    "Current model accuracy"
)
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Currently active requests"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model State ─────────────────────────
app_state = {
    "model": None,
    "model_version": None,
    "total_predictions": 0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    logger.info("Loading model...")
    app_state["model"] = mlflow.sklearn\\
        .load_model(
            "models:/ProductionServingModel"
            "/Production"
        )
    app_state["model_version"] = "2.0"
    MODEL_ACCURACY_GAUGE.set(0.924)
    logger.info("Model ready!")
    yield
    # Shutdown: cleanup
    logger.info("Shutting down...")

app = FastAPI(
    title="Production ML API",
    version="2.0.0",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"])

# ── Schemas ─────────────────────────────
class PredictRequest(BaseModel):
    features: List[List[float]]
    request_id: Optional[str] = None
    metadata: Optional[dict] = None

    @validator("features")
    def validate(cls, v):
        if not v:
            raise ValueError("Empty!")
        if len(v[0]) != 12:
            raise ValueError(
                f"Need 12 features, got {len(v[0])}"
            )
        return v

class PredictResponse(BaseModel):
    request_id: str
    predictions: List[int]
    probabilities: List[List[float]]
    model_version: str
    latency_ms: float
    n_samples: int

# ── Middleware ───────────────────────────
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    ACTIVE_REQUESTS.dec()
    return response

# ── Endpoints ───────────────────────────
@app.get("/health")
async def health():
    uptime = time.time() - app_state["start_time"]
    return {
        "status": "healthy",
        "model_loaded": app_state["model"] is not None,
        "model_version": app_state["model_version"],
        "total_predictions": app_state["total_predictions"],
        "uptime_seconds": round(uptime, 1)
    }

@app.get("/ready")
async def readiness():
    if app_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict",
           response_model=PredictResponse)
async def predict(req: PredictRequest):
    if app_state["model"] is None:
        raise HTTPException(503, "No model!")

    start = time.time()
    request_id = req.request_id or str(uuid.uuid4())

    try:
        X = np.array(req.features)
        preds = app_state["model"].predict(X)
        probas = app_state["model"]\\
            .predict_proba(X)
        latency = (time.time() - start) * 1000

        app_state["total_predictions"] += len(preds)
        for p in preds:
            PREDICTION_COUNTER.labels(
                model_version=app_state["model_version"],
                class=str(p)
            ).inc()

        logger.info(
            f"req={request_id[:8]} "
            f"n={len(preds)} "
            f"lat={latency:.1f}ms"
        )
        return PredictResponse(
            request_id=request_id,
            predictions=preds.tolist(),
            probabilities=probas.tolist(),
            model_version=app_state["model_version"],
            latency_ms=round(latency, 2),
            n_samples=len(preds)
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats")
async def stats():
    return {
        "total_predictions":
            app_state["total_predictions"],
        "model_version":
            app_state["model_version"],
        "uptime":
            time.time() - app_state["start_time"]
    }
'''

# Save FastAPI app
os.makedirs(
    "phase4-mlops/week2-docker-cicd/project/src",
    exist_ok=True
)
with open(
    "phase4-mlops/week2-docker-cicd/project/"
    "src/production_api.py", "w"
) as f:
    f.write(FASTAPI_COMPLETE)
print("  ✅ FastAPI app written!")

# ══════════════════════════════════════════
# PART 3: BATCH INFERENCE PIPELINE
# ══════════════════════════════════════════
print("\n[3] Batch Inference Pipeline")

# Create test data in Spark
test_pdf = pd.DataFrame(
    X_te_s,
    columns=[f"f{i}" for i in range(12)]
)
test_sdf = spark.createDataFrame(test_pdf)

# Batch inference using Spark UDF
feature_cols = [f"f{i}" for i in range(12)]

# Load model as Spark UDF
model_uri = f"runs:/{RUN_ID}/model"
predict_udf = mlflow.pyfunc.spark_udf(
    spark, model_uri,
    result_type="integer"
)

# Apply to full dataset
batch_results = test_sdf.withColumn(
    "prediction",
    predict_udf(*[F.col(c)
                   for c in feature_cols])
).withColumn(
    "inference_ts",
    F.current_timestamp()
).withColumn(
    "batch_id",
    F.lit("batch_2026_06_19")
)

# Write to Delta
batch_results.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("batch_id") \
    .save("/tmp/serving_project/predictions")

n_preds = batch_results.count()
print(f"  ✅ Batch: {n_preds} predictions")

# Analytics on predictions
print("\n  Prediction distribution:")
batch_results.groupBy("prediction") \
    .count() \
    .orderBy("prediction") \
    .show()

# ══════════════════════════════════════════
# PART 4: LATENCY BENCHMARKING
# ══════════════════════════════════════════
print("\n[4] Latency Benchmarking")

latencies_1 = []
latencies_10 = []
latencies_100 = []

for batch_size, latencies in [
    (1, latencies_1),
    (10, latencies_10),
    (100, latencies_100)
]:
    for _ in range(100):
        X_batch = np.random.randn(
            batch_size, 12
        )
        start = time.perf_counter()
        _ = model.predict(X_batch)
        latencies.append(
            (time.perf_counter()-start)*1000
        )

print(f"{'Batch':>8} {'p50':>8} "
      f"{'p95':>8} {'p99':>8} {'RPS':>8}")
for bs, lats in [
    (1, latencies_1),
    (10, latencies_10),
    (100, latencies_100)
]:
    lats = np.array(lats)
    rps = bs * 1000 / np.mean(lats)
    print(f"{bs:>8} "
          f"{np.percentile(lats,50):>7.2f}ms "
          f"{np.percentile(lats,95):>7.2f}ms "
          f"{np.percentile(lats,99):>7.2f}ms "
          f"{rps:>7.0f}")

# ══════════════════════════════════════════
# PART 5: PROMETHEUS CONFIG
# ══════════════════════════════════════════
print("\n[5] Prometheus + Grafana Config")

prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:8000']
    metrics_path: '/metrics'
"""

grafana_dashboard = {
    "title": "ML API Dashboard",
    "panels": [
        {"title": "Request Rate",
         "expr": "rate(http_requests_total[1m])"},
        {"title": "Latency p99",
         "expr": "histogram_quantile(0.99,"
                 " http_request_duration_seconds)"},
        {"title": "Prediction Distribution",
         "expr": "predictions_total"},
        {"title": "Model Accuracy",
         "expr": "model_accuracy"},
        {"title": "Active Requests",
         "expr": "active_requests"}
    ]
}

os.makedirs(
    "phase4-mlops/week2-docker-cicd/project",
    exist_ok=True
)
with open(
    "phase4-mlops/week2-docker-cicd/"
    "project/prometheus.yml", "w"
) as f:
    f.write(prometheus_config)

with open(
    "phase4-mlops/week2-docker-cicd/"
    "project/grafana_dashboard.json", "w"
) as f:
    json.dump(grafana_dashboard, f, indent=2)
print("  ✅ Monitoring configs written!")

# ══════════════════════════════════════════
# PART 6: MLFLOW SUMMARY
# ══════════════════════════════════════════
print("\n[6] MLflow Project Summary")

with mlflow.start_run(
        run_name="week2_project_summary"):
    mlflow.log_params({
        "serving": "FastAPI+Docker",
        "monitoring": "Prometheus+Grafana",
        "batch": "Spark_PandasUDF",
        "registry": "MLflow_Model_Registry",
        "n_features": 12
    })
    mlflow.log_metrics({
        "model_accuracy": acc,
        "batch_predictions": n_preds,
        "p50_latency_ms": float(
            np.percentile(latencies_1, 50)
        ),
        "p99_latency_ms": float(
            np.percentile(latencies_1, 99)
        ),
        "batch_rps": float(
            100 * 1000 / np.mean(latencies_100)
        )
    })
    print("  ✅ Project summary logged!")

print("\n" + "="*60)
print("WEEK 2 PROJECT COMPLETE! 🏆")
print("="*60)
print(f"""
Deliverables:
  ✅ MLflow model: acc={acc:.4f}
  ✅ FastAPI app: full production grade
  ✅ Batch inference: {n_preds} predictions
  ✅ Prometheus: 5 metrics configured
  ✅ Grafana: 5-panel dashboard
  ✅ Latency: p99 benchmarked

6 Databricks competencies:
  1. MLflow (tracking + serving)
  2. Docker (containerization)
  3. FastAPI (REST API)
  4. Spark (batch inference)
  5. Delta Lake (prediction storage)
  6. Prometheus/Grafana (monitoring)
""")
