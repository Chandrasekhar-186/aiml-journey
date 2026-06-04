# Phase 4 Day 1 — MLflow Production Serving
# Date: June 12, 2026
# Taking Phase 3 models to production!

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

print("="*60)
print("MLflow Production — Model Serving")
print("="*60)

"""
MLFLOW PRODUCTION STACK:

Training (Phase 1-3): mlflow.log_*
                       mlflow.register_model
Serving (Phase 4):    REST endpoint!
                       mlflow models serve
                       Databricks Model Serving

Flow:
Train → Log → Register → Staging →
Test → Production → Serve → Monitor

Serving options:
1. mlflow models serve (local REST API)
2. Databricks Model Serving (managed!)
3. Custom FastAPI wrapper
4. Batch inference on Spark

REST endpoint:
POST /invocations
Content-Type: application/json
{"inputs": [[feat1, feat2, ...]]}
→ {"predictions": [0, 1, 1, 0]}
"""

client = MlflowClient()

# 1. Train + register a production model
print("\n=== TRAINING + REGISTERING MODEL ===")

X, y = make_classification(
    n_samples=2000, n_features=20,
    n_informative=12, random_state=42
)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2,
                     random_state=42)

mlflow.set_experiment("phase4_model_serving")

with mlflow.start_run(
        run_name="RF_production_candidate"):

    # Train
    model = RandomForestClassifier(
        n_estimators=100, random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log everything
    mlflow.log_params({
        "n_estimators": 100,
        "model_type": "RandomForest"
    })
    mlflow.log_metric("accuracy", acc)

    # Infer signature (critical for serving!)
    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )
    print(f"Model signature: {signature}")

    # Log with signature
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=
            "ProductionClassifier"
    )

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {acc:.4f}")

# 2. Model lifecycle management
print("\n=== MODEL LIFECYCLE ===")
print("""
Model Stages:
None:       just registered
Staging:    testing + validation
Production: serving live traffic
Archived:   retired model

Transition commands:
client.transition_model_version_stage(
    name="ProductionClassifier",
    version=1,
    stage="Staging"
)
client.transition_model_version_stage(
    name="ProductionClassifier",
    version=1,
    stage="Production"
)

Load production model:
model = mlflow.sklearn.load_model(
    "models:/ProductionClassifier/Production"
)
""")

# 3. Custom pyfunc model
print("\n=== CUSTOM PYFUNC MODEL ===")

class PreprocessingModel(mlflow.pyfunc.PythonModel):
    """
    Custom model with preprocessing baked in!
    This is how you ship preprocessing + model
    as ONE artifact — no training-serving skew!
    """
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input):
        # model_input: pandas DataFrame
        X = model_input.values
        # Apply same preprocessing as training!
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(
            X_scaled
        )
        probabilities = self.model.predict_proba(
            X_scaled
        )
        return pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities[:, 1],
            "confidence": np.max(
                probabilities, axis=1
            )
        })

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
rf_scaled = RandomForestClassifier(
    n_estimators=50, random_state=42
)
rf_scaled.fit(X_scaled, y_train)

custom_model = PreprocessingModel(
    rf_scaled, scaler
)

# Log custom model
with mlflow.start_run(
        run_name="custom_pyfunc_model"):
    input_example = pd.DataFrame(
        X_test[:3],
        columns=[f"f{i}" for i in range(20)]
    )
    mlflow.pyfunc.log_model(
        artifact_path="preprocessing_model",
        python_model=custom_model,
        input_example=input_example,
        registered_model_name=
            "PreprocessingClassifier"
    )

    # Test prediction
    result = custom_model.predict(
        None, input_example
    )
    print("Custom model predictions:")
    print(result)

# 4. REST API serving
print("\n=== REST API SERVING ===")
print("""
# Serve locally:
mlflow models serve \\
    -m "models:/ProductionClassifier/Production" \\
    -p 5001 \\
    --no-conda

# Call the endpoint:
import requests
import json

data = {"inputs": X_test[:3].tolist()}
response = requests.post(
    "http://localhost:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)
predictions = response.json()["predictions"]

# On Databricks:
# Create endpoint in Model Serving UI
# Or programmatically:
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
endpoint = w.serving_endpoints.create(
    name="my-model-endpoint",
    config=ServedModelInput(
        model_name="ProductionClassifier",
        model_version="1",
        workload_size="Small",
        scale_to_zero_enabled=True
    )
)
""")

# 5. A/B testing for ML models
print("\n=== A/B TESTING FOR ML ===")

class ABTestingModel(mlflow.pyfunc.PythonModel):
    """
    Routes traffic between model A and model B
    Based on percentage split!
    """
    def __init__(self, model_a, model_b,
                  traffic_split=0.8):
        self.model_a = model_a
        self.model_b = model_b
        self.split = traffic_split
        self.a_count = 0
        self.b_count = 0

    def predict(self, context, model_input):
        X = model_input.values
        results = []
        models_used = []

        for i in range(len(X)):
            # Route to A or B
            if np.random.random() < self.split:
                pred = self.model_a.predict(
                    X[i:i+1]
                )[0]
                models_used.append("model_A")
                self.a_count += 1
            else:
                pred = self.model_b.predict(
                    X[i:i+1]
                )[0]
                models_used.append("model_B")
                self.b_count += 1
            results.append(pred)

        return pd.DataFrame({
            "prediction": results,
            "model_version": models_used
        })

    def get_traffic_stats(self):
        total = self.a_count + self.b_count
        return {
            "model_A_traffic":
                self.a_count / max(total, 1),
            "model_B_traffic":
                self.b_count / max(total, 1),
            "total_requests": total
        }

# Train two model versions
model_v1 = RandomForestClassifier(
    n_estimators=50, random_state=42
)
model_v2 = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model_v1.fit(X_train, y_train)
model_v2.fit(X_train, y_train)

ab_model = ABTestingModel(
    model_v1, model_v2,
    traffic_split=0.8  # 80% → v1, 20% → v2
)

# Simulate traffic
test_df = pd.DataFrame(
    X_test[:100],
    columns=[f"f{i}" for i in range(20)]
)
preds = ab_model.predict(None, test_df)
stats = ab_model.get_traffic_stats()

print(f"A/B Test Results:")
print(f"  Model A traffic: "
      f"{stats['model_A_traffic']:.1%}")
print(f"  Model B traffic: "
      f"{stats['model_B_traffic']:.1%}")
print(f"  Total requests:  "
      f"{stats['total_requests']}")

# Log A/B experiment
mlflow.set_experiment("phase4_ab_testing")
with mlflow.start_run(
        run_name="AB_test_v1_v2"):
    mlflow.log_params({
        "model_a": "RF_n50",
        "model_b": "RF_n100",
        "split": "80/20"
    })
    mlflow.log_metrics({
        "model_a_traffic":
            stats['model_A_traffic'],
        "model_b_traffic":
            stats['model_B_traffic'],
        "model_a_acc":
            accuracy_score(
                y_test,
                model_v1.predict(X_test)
            ),
        "model_b_acc":
            accuracy_score(
                y_test,
                model_v2.predict(X_test)
            )
    })
    print("\nA/B test logged to MLflow!")

print("\n" + "="*60)
print("MLflow Production Serving — MASTERED!")
print("Phase 4 Day 1 COMPLETE! 🏭")
print("="*60)
