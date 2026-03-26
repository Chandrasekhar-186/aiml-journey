# Day 11 — MLflow Model Registry
# Date: March 23, 2026
# This is how Databricks manages models in production!

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. Register a model from previous run
# (get run_id from mlflow ui)
model_name = "BreastCancerClassifier"

# Register model
mlflow.register_model(
    f"runs:/<YOUR_RUN_ID>/random_forest",
    model_name
)

# 2. Transition model stages
# Staging → Production workflow!
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"  # None → Staging → Production
)
print(f"Model moved to Staging!")

# 3. Add description
client.update_registered_model(
    name=model_name,
    description="Random Forest for cancer detection. "
                "Trained on breast cancer dataset. "
                "F1: 0.967, ROC AUC: 0.994"
)

# 4. Load model from registry
loaded_model = mlflow.sklearn.load_model(
    f"models:/{model_name}/Staging"
)
print(f"Model loaded from registry: {loaded_model}")

# 5. List all registered models
for rm in client.search_registered_models():
    print(f"Model: {rm.name}")
    for mv in rm.latest_versions:
        print(f"  v{mv.version} — {mv.current_stage}")
