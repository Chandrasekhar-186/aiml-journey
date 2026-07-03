# Phase 4 Day 9 — ML Observability + Cost
# Date: June 20, 2026
# Know WHY before users know WHAT!

import numpy as np
import pandas as pd
import mlflow
import logging
import json
import time
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Observability") \
    .getOrCreate()

print("="*60)
print("ML Observability + Cost Optimization")
print("="*60)

"""
THE THREE PILLARS OF OBSERVABILITY:

1. LOGS — what happened?
   → Structured logs (JSON)
   → Every prediction logged
   → Error traces with context
   → Queryable in Databricks Log Analytics

2. METRICS — how fast/much?
   → Latency percentiles (p50/p95/p99)
   → Throughput (requests/second)
   → Error rate (%)
   → Model accuracy over time
   → Feature drift scores

3. TRACES — why did it happen?
   → End-to-end request flow
   → Per-step timing
   → MLflow tracing (new in 2.13+!)
   → Distributed request correlation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRUCTURED LOGGING FOR ML:

Wrong (unstructured):
  print(f"Predicted {pred} in {latency}ms")

Right (structured JSON):
  logger.info({
    "event": "prediction",
    "request_id": "abc-123",
    "model_version": "2.0",
    "prediction": 1,
    "confidence": 0.94,
    "latency_ms": 23.4,
    "feature_hash": "d4f2...",
    "timestamp": "2026-06-20T10:23:45Z"
  })

Why structured?
→ Queryable: WHERE confidence < 0.5
→ Aggregatable: AVG(latency_ms) by hour
→ Alertable: COUNT errors > 100/min
→ Auditable: full prediction history

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLFLOW TRACING (Databricks native!):

@mlflow.trace
def predict_with_trace(features):
    with mlflow.start_span("preprocessing"):
        X = preprocess(features)

    with mlflow.start_span("inference") as span:
        pred = model.predict(X)
        span.set_attribute("n_samples", len(pred))

    with mlflow.start_span("postprocessing"):
        result = format_output(pred)

    return result

→ Every span timed automatically
→ View in MLflow UI → Traces tab
→ See exactly where time is spent!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COST OPTIMIZATION FOR DATABRICKS ML:

Cluster costs = biggest expense!

Strategies:
1. Right-size clusters:
   Don't use 32-node cluster for dev!
   Dev: 2-4 workers
   Prod: auto-scale 4-16 workers

2. Spot instances (preemptible):
   70-90% cheaper than on-demand
   Risk: can be terminated
   Use for: batch jobs, training
   Don't use for: streaming, online serving

3. Photon engine:
   Free with Databricks
   10x faster for SQL + Delta
   Enable: --conf spark.databricks.photon=true

4. Delta caching:
   Cache hot data in SSD
   Avoid re-reading from S3 repeatedly
   Enable: spark.databricks.io.cache.enabled

5. Partition pruning:
   Write data partitioned by date/category
   Queries scan ONLY needed partitions
   10x less data read = 10x cheaper!

6. Auto-terminate clusters:
   Set: auto terminate after 30 min idle
   Never leave clusters running overnight!

7. Serverless compute (new!):
   Pay per second, no cluster management
   Best for: notebooks, short jobs
   Databricks Serverless SQL Warehouses
"""

# 1. Structured logging system
print("\n=== STRUCTURED LOGGING ===")

class MLLogger:
    """Production structured logger for ML"""

    def __init__(self, service: str,
                  model_version: str):
        self.service = service
        self.model_version = model_version

        # Configure JSON logger
        self.logger = logging.getLogger(service)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _log(self, level: str,
               event: str, **kwargs):
        record = {
            "timestamp":
                datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service,
            "model_version": self.model_version,
            "event": event,
            **kwargs
        }
        self.logger.info(json.dumps(record))
        return record

    def prediction(self, request_id: str,
                    n_samples: int,
                    prediction: list,
                    confidence: list,
                    latency_ms: float):
        return self._log(
            "INFO", "prediction",
            request_id=request_id,
            n_samples=n_samples,
            predictions=prediction,
            avg_confidence=np.mean(confidence),
            latency_ms=round(latency_ms, 2)
        )

    def error(self, request_id: str,
               error_type: str,
               message: str):
        return self._log(
            "ERROR", "prediction_error",
            request_id=request_id,
            error_type=error_type,
            message=message
        )

    def drift_alert(self, feature: str,
                     ks_pvalue: float,
                     psi: float):
        return self._log(
            "WARN", "drift_detected",
            feature=feature,
            ks_pvalue=round(ks_pvalue, 4),
            psi=round(psi, 4),
            action="retraining_triggered"
        )

    def model_loaded(self, load_time_ms: float):
        return self._log(
            "INFO", "model_loaded",
            load_time_ms=round(load_time_ms, 1),
            status="ready"
        )

# Test logging
ml_logger = MLLogger(
    "ml-prediction-service", "2.0"
)

print("Sample structured logs:")
ml_logger.model_loaded(342.5)
ml_logger.prediction(
    "req-abc-123", 5,
    [0, 1, 1, 0, 1],
    [0.92, 0.87, 0.95, 0.78, 0.91],
    23.4
)
ml_logger.drift_alert("feature_1", 0.003, 0.31)
ml_logger.error(
    "req-xyz-789",
    "ValidationError",
    "Expected 12 features, got 10"
)

# 2. SLA monitoring
print("\n=== SLA MONITORING ===")

class SLAMonitor:
    """Monitor SLA compliance"""

    def __init__(self,
                  p99_threshold_ms: float = 200,
                  error_rate_threshold: float = 0.01,
                  accuracy_threshold: float = 0.85):
        self.p99_threshold = p99_threshold_ms
        self.error_threshold = error_rate_threshold
        self.accuracy_threshold = accuracy_threshold
        self.latencies = []
        self.errors = 0
        self.total = 0

    def record(self, latency_ms: float,
                success: bool):
        self.latencies.append(latency_ms)
        self.total += 1
        if not success:
            self.errors += 1

    def check_sla(self) -> dict:
        if not self.latencies:
            return {"status": "no_data"}

        p99 = np.percentile(self.latencies, 99)
        error_rate = self.errors / max(
            self.total, 1
        )

        sla_violations = []
        if p99 > self.p99_threshold:
            sla_violations.append(
                f"p99 {p99:.1f}ms > "
                f"{self.p99_threshold}ms"
            )
        if error_rate > self.error_threshold:
            sla_violations.append(
                f"error rate {error_rate:.1%} > "
                f"{self.error_threshold:.1%}"
            )

        return {
            "p50_ms": np.percentile(
                self.latencies, 50
            ),
            "p99_ms": p99,
            "error_rate": error_rate,
            "total_requests": self.total,
            "sla_violated": len(
                sla_violations
            ) > 0,
            "violations": sla_violations
        }

# Simulate traffic
sla = SLAMonitor(p99_threshold_ms=100)
np.random.seed(42)
for _ in range(1000):
    latency = np.random.lognormal(3, 0.5)
    success = np.random.random() > 0.005
    sla.record(latency, success)

result = sla.check_sla()
print(f"SLA Report:")
print(f"  p50:         {result['p50_ms']:.1f}ms")
print(f"  p99:         {result['p99_ms']:.1f}ms")
print(f"  Error rate:  {result['error_rate']:.2%}")
print(f"  SLA violated: {result['sla_violated']}")
if result['violations']:
    print(f"  Violations: {result['violations']}")

# 3. Cost analysis
print("\n=== COST OPTIMIZATION ANALYSIS ===")

cost_scenarios = {
    "On-demand (current)": {
        "instance": "i3.2xlarge",
        "workers": 8,
        "$/hr": 0.624,
        "hours/day": 8,
        "days/month": 22
    },
    "Spot instances": {
        "instance": "i3.2xlarge (spot)",
        "workers": 8,
        "$/hr": 0.187,  # ~70% cheaper
        "hours/day": 8,
        "days/month": 22
    },
    "Right-sized + Spot": {
        "instance": "m5.xlarge (spot)",
        "workers": 4,  # right-sized!
        "$/hr": 0.063,
        "hours/day": 8,
        "days/month": 22
    }
}

print(f"{'Scenario':25} {'Monthly $':>12} "
      f"{'Savings':>10}")
print("-" * 50)

base_cost = None
for name, config in cost_scenarios.items():
    monthly = (
        config["$/hr"] *
        config["workers"] *
        config["hours/day"] *
        config["days/month"]
    )
    if base_cost is None:
        base_cost = monthly
        savings = "baseline"
    else:
        pct = (base_cost-monthly)/base_cost
        savings = f"{pct:.0%} cheaper"
    print(f"{name:25} ${monthly:>10.0f} "
          f"{savings:>10}")

# 4. Partition pruning impact
print("\n=== PARTITION PRUNING IMPACT ===")

# Simulate data with/without partitioning
data = [
    (f"2026-{m:02d}-{d:02d}",
     float(i % 100) / 100,
     i % 10)
    for i, (m, d) in enumerate([
        (m, d)
        for m in range(1, 7)
        for d in range(1, 29)
    ])
]

df = spark.createDataFrame(
    data, ["date", "value", "category"]
)

# Write WITHOUT partitioning
df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/unpartitioned")

# Write WITH partitioning
df.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("date") \
    .save("/tmp/partitioned")

# Query: last 7 days
start_date = "2026-06-01"
print(f"Query: date >= {start_date}")
print("Without partitioning: scans ALL data!")
print("With partitioning: scans only June!")
print("Impact: ~6x less data read → 6x cheaper!")

# 5. MLflow cost tracking
mlflow.set_experiment("phase4_observability")
with mlflow.start_run(
        run_name="observability_cost"):
    mlflow.log_params({
        "logging": "structured_JSON",
        "tracing": "MLflow_spans",
        "monitoring": "Prometheus_SLA",
        "compute": "spot_instances"
    })
    mlflow.log_metrics({
        "p99_latency_ms": result['p99_ms'],
        "error_rate": result['error_rate'],
        "sla_violated":
            int(result['sla_violated']),
        "monthly_cost_savings_pct": 0.70
    })
    print("\nObservability metrics logged!")

print("\n" + "="*60)
print("ML Observability + Cost — MASTERED! 📊")
print("Phase 4 Day 9 COMPLETE!")
print("="*60)
