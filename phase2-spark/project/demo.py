# ============================================
# ML Model Monitor — Quick Demo
# Run this to see the full pipeline!
# ============================================

import subprocess
import sys

print("="*60)
print("ML Model Monitor — Full Demo")
print("="*60)
print()
print("This demo showcases:")
print("✅ Bronze: Streaming ingestion")
print("✅ Silver: Data quality + enrichment")
print("✅ Gold:   Metrics + drift detection")
print("✅ MLflow: Experiment tracking")
print("✅ GraphFrames: Model dependency graph")
print()
print("Architecture:")
print("Kafka → Spark Streaming → Bronze Delta")
print("     → Silver Delta → Gold Delta")
print("     → MLflow Registry → Alerts")
print()
print("To run full pipeline:")
print("1. python bronze_layer.py")
print("2. python silver_gold_layers.py")
print("3. python mlflow_integration.py")
print("4. mlflow ui  # view at localhost:5000")
print()
print("Key metrics this system tracks:")
print("→ Model accuracy per prediction batch")
print("→ Z-score drift detection (|z|>2.5=alert)")
print("→ Latency buckets (fast/medium/slow)")
print("→ Anomaly rate per model")
print()
print("Databricks competencies demonstrated:")
competencies = [
    "Apache Spark (Structured Streaming)",
    "Delta Lake (Bronze/Silver/Gold Lakehouse)",
    "MLflow (tracking + model registry)",
    "GraphFrames (model dependency PageRank)",
    "Kafka integration (event streaming)",
    "Production patterns (checkpoints, MERGE)",
    "Drift detection (statistical Z-score)"
]
for i, c in enumerate(competencies, 1):
    print(f"{i}. {c}")
