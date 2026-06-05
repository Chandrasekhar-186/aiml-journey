# Phase 4 Day 2 — Feature Stores + Monitoring
# Date: June 13, 2026
# Keeping models healthy after deployment!

import numpy as np
import pandas as pd
from scipy import stats
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable

spark = SparkSession.builder \
    .appName("FeatureStoreMonitoring") \
    .getOrCreate()

print("="*60)
print("Feature Stores + ML Monitoring")
print("="*60)

"""
FEATURE STORE — WHY IT MATTERS

Problem without feature store:
Training pipeline:  compute features A, B, C
Serving pipeline:   compute features A, B, C
→ Two separate implementations!
→ Tiny difference = TRAINING-SERVING SKEW
→ Model trained on one thing, served another
→ Silent accuracy degradation!

Feature store solution:
Single feature computation → stored once
Training reads from feature store
Serving reads SAME feature store
→ Guaranteed consistency! ✅

Databricks Feature Store:
→ Delta Lake backed (ACID + time travel!)
→ Point-in-time lookups (no data leakage!)
→ Automatic lineage tracking
→ Works with MLflow Model Registry

Key concepts:
Feature table:    Delta table of computed features
Training set:     historical features + labels
Online store:     low-latency serving features
Offline store:    batch training features
Point-in-time:    features AT training time only!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ML MONITORING — THE SILENT KILLER

Models decay silently:
Week 1:  accuracy = 92%
Month 1: accuracy = 88% (slight drop)
Month 3: accuracy = 71% (serious!)
Month 6: accuracy = 54% (barely better than random!)

Why? Data drift! The world changes:
→ User behavior changes
→ Seasonal patterns shift
→ Product features added/removed
→ Economic conditions change

Types of drift:
Data drift (covariate shift):
  P(X) changes but P(Y|X) same
  "Feature distribution changed"

Label drift (prior probability shift):
  P(Y) changes
  "Class balance changed"

Concept drift:
  P(Y|X) changes
  "The relationship changed!"
  Most dangerous! Model fundamentally wrong.

Detection methods:
Statistical tests on feature distributions
→ KS test: continuous features
→ Chi-square: categorical features
→ PSI: Population Stability Index
→ Jensen-Shannon divergence

Performance monitoring:
→ Track accuracy over time
→ Alert when drops below threshold
→ Compare prediction distribution changes
"""

# 1. Simulated Feature Store
print("\n=== SIMULATED FEATURE STORE ===")

class DeltaFeatureStore:
    """
    Simulates Databricks Feature Store
    backed by Delta Lake
    """
    def __init__(self, spark):
        self.spark = spark
        self.tables = {}

    def create_feature_table(
        self, name: str,
        df,
        primary_keys: list,
        timestamp_key: str = None
    ):
        """Create or update feature table"""
        # Write to Delta Lake
        path = f"/tmp/feature_store/{name}"
        df.write.format("delta") \
            .mode("overwrite") \
            .save(path)
        self.tables[name] = {
            "path": path,
            "primary_keys": primary_keys,
            "timestamp_key": timestamp_key
        }
        print(f"Feature table '{name}' created!")
        print(f"  Rows: {df.count()}")
        print(f"  Keys: {primary_keys}")

    def get_table(self, name: str):
        if name not in self.tables:
            raise ValueError(
                f"Table {name} not found!"
            )
        return self.spark.read.format("delta") \
            .load(self.tables[name]["path"])

    def create_training_set(
        self, observation_df,
        feature_lookups: list,
        label_col: str
    ):
        """
        Join features to observations
        Point-in-time safe!
        """
        result = observation_df
        for lookup in feature_lookups:
            feat_df = self.get_table(
                lookup["table"]
            )
            result = result.join(
                feat_df.select(
                    lookup["lookup_key"],
                    *lookup["feature_names"]
                ),
                on=lookup["lookup_key"],
                how="left"
            )
        return result

# Create feature tables
fs = DeltaFeatureStore(spark)

# User features
user_features = spark.createDataFrame([
    (i, float(i%100)/100,
     float(i%50)/50,
     i%10,
     f"segment_{i%5}")
    for i in range(1000)
], ["user_id", "avg_session_time",
    "purchase_rate", "tenure_months",
    "user_segment"])

fs.create_feature_table(
    "user_features",
    user_features,
    primary_keys=["user_id"]
)

# Item features
item_features = spark.createDataFrame([
    (i, float(i%100)/100,
     i%20, float(i%5)/10)
    for i in range(500)
], ["item_id", "popularity_score",
    "category_id", "avg_rating"])

fs.create_feature_table(
    "item_features",
    item_features,
    primary_keys=["item_id"]
)

print("\nFeature Store contents:")
print(f"  user_features: "
      f"{fs.get_table('user_features').count()} rows")
print(f"  item_features: "
      f"{fs.get_table('item_features').count()} rows")

# 2. Statistical drift detection
print("\n=== DRIFT DETECTION ===")

class DriftDetector:
    """Production drift detection"""

    def ks_test(self, reference: np.ndarray,
                  current: np.ndarray,
                  threshold: float = 0.05
                  ) -> dict:
        """
        Kolmogorov-Smirnov test for
        continuous feature drift
        """
        statistic, p_value = \
            stats.ks_2samp(reference, current)
        return {
            "test": "KS",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": p_value < threshold,
            "severity": "high" if p_value < 0.01
                        else "medium" if
                        p_value < threshold
                        else "none"
        }

    def psi(self, reference: np.ndarray,
             current: np.ndarray,
             buckets: int = 10) -> dict:
        """
        Population Stability Index
        PSI < 0.1:  no shift
        PSI < 0.2:  slight shift
        PSI >= 0.2: significant shift!
        """
        # Create buckets from reference
        breakpoints = np.percentile(
            reference,
            np.linspace(0, 100, buckets+1)
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(
            reference, bins=breakpoints
        )[0]
        cur_counts = np.histogram(
            current, bins=breakpoints
        )[0]

        # Avoid zeros
        ref_pct = (ref_counts + 0.0001) / \
                   len(reference)
        cur_pct = (cur_counts + 0.0001) / \
                   len(current)

        psi_val = np.sum(
            (cur_pct - ref_pct) *
            np.log(cur_pct / ref_pct)
        )

        return {
            "test": "PSI",
            "psi_value": float(psi_val),
            "drifted": psi_val >= 0.2,
            "severity":
                "high" if psi_val >= 0.25 else
                "medium" if psi_val >= 0.1 else
                "none"
        }

    def detect_all(self,
                    reference_df: pd.DataFrame,
                    current_df: pd.DataFrame,
                    features: list) -> dict:
        """Run all drift tests on features"""
        results = {}
        for feat in features:
            ref = reference_df[feat].dropna()\
                      .values
            cur = current_df[feat].dropna()\
                      .values

            ks = self.ks_test(ref, cur)
            psi_r = self.psi(ref, cur)

            results[feat] = {
                "ks": ks,
                "psi": psi_r,
                "alert":
                    ks["drifted"] or
                    psi_r["drifted"]
            }
        return results

# Generate reference + drifted data
np.random.seed(42)
n = 1000

reference_data = pd.DataFrame({
    "feature_1": np.random.normal(0, 1, n),
    "feature_2": np.random.exponential(2, n),
    "feature_3": np.random.uniform(0, 10, n),
})

# Simulate drift: mean shifted
drifted_data = pd.DataFrame({
    "feature_1": np.random.normal(
        1.5, 1, n),  # mean shifted!
    "feature_2": np.random.exponential(
        2, n),       # no drift
    "feature_3": np.random.uniform(
        3, 13, n),   # range shifted!
})

detector = DriftDetector()
drift_results = detector.detect_all(
    reference_data, drifted_data,
    features=["feature_1", "feature_2",
               "feature_3"]
)

print("Drift Detection Results:")
alerts = []
for feat, result in drift_results.items():
    ks = result["ks"]
    psi = result["psi"]
    alert = "🚨 DRIFT!" if result["alert"] \
            else "✅ OK"
    print(f"\n  {feat}: {alert}")
    print(f"    KS p-value: {ks['p_value']:.4f}"
          f" ({ks['severity']})")
    print(f"    PSI value:  {psi['psi_value']:.4f}"
          f" ({psi['severity']})")
    if result["alert"]:
        alerts.append(feat)

print(f"\n🚨 Features with drift: {alerts}")

# 3. Model performance monitoring
print("\n=== MODEL PERFORMANCE MONITORING ===")

class ModelMonitor:
    def __init__(self, model_name: str,
                  threshold: float = 0.85):
        self.model_name = model_name
        self.threshold = threshold
        self.history = []

    def log_batch(self, y_true, y_pred,
                   timestamp: str):
        from sklearn.metrics import (
            accuracy_score, f1_score
        )
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred,
                       average='weighted')
        self.history.append({
            "timestamp": timestamp,
            "accuracy": acc,
            "f1": f1,
            "n_samples": len(y_true),
            "alert": acc < self.threshold
        })
        return acc, f1

    def check_alerts(self):
        if not self.history:
            return []
        recent = self.history[-1]
        alerts = []
        if recent["alert"]:
            alerts.append(
                f"⚠️ {self.model_name}: "
                f"accuracy {recent['accuracy']:.3f}"
                f" below threshold "
                f"{self.threshold}"
            )
        # Check for degradation trend
        if len(self.history) >= 3:
            recent_accs = [
                h["accuracy"]
                for h in self.history[-3:]
            ]
            if all(
                recent_accs[i] > recent_accs[i+1]
                for i in range(len(recent_accs)-1)
            ):
                alerts.append(
                    f"📉 {self.model_name}: "
                    f"consistent degradation trend!"
                )
        return alerts

# Simulate model degradation over time
monitor = ModelMonitor(
    "ProductionClassifier", threshold=0.85
)
np.random.seed(42)
timestamps = [
    "2026-06-01", "2026-06-08",
    "2026-06-15", "2026-06-22"
]
base_accuracies = [0.91, 0.88, 0.84, 0.79]

print("Model Performance Over Time:")
for ts, base_acc in zip(timestamps,
                          base_accuracies):
    y_true = np.random.randint(0, 2, 200)
    noise = np.random.normal(0, 0.05)
    y_pred = (np.random.random(200) 
               base_acc + noise).astype(int)
    acc, f1 = monitor.log_batch(
        y_true, y_pred, ts
    )
    status = "🚨" if acc < 0.85 else "✅"
    print(f"  {ts}: acc={acc:.3f} "
          f"f1={f1:.3f} {status}")

alerts = monitor.check_alerts()
if alerts:
    print(f"\nAlerts generated:")
    for alert in alerts:
        print(f"  {alert}")

# 4. Log everything to MLflow
mlflow.set_experiment("phase4_monitoring")
with mlflow.start_run(
        run_name="drift_monitoring"):
    # Log drift results
    for feat, result in drift_results.items():
        mlflow.log_metric(
            f"{feat}_ks_pval",
            result["ks"]["p_value"]
        )
        mlflow.log_metric(
            f"{feat}_psi",
            result["psi"]["psi_value"]
        )

    mlflow.log_metric(
        "drifted_features", len(alerts)
    )
    mlflow.log_param(
        "drift_tests", "KS+PSI"
    )

    # Log model performance history
    for h in monitor.history:
        mlflow.log_metric(
            "accuracy",
            h["accuracy"],
            step=monitor.history.index(h)
        )

    print("\nMonitoring logged to MLflow!")

print("\n" + "="*60)
print("Feature Stores + Monitoring — MASTERED!")
print("Phase 4 Day 2 COMPLETE! 🏭")
print("="*60)
