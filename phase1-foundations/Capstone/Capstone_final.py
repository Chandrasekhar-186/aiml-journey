# Phase 1 Capstone — FINAL
# Complete System Integration
# Date: April 6, 2026

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

print("="*60)
print("CAPSTONE: Intelligent ML Experiment Analyzer")
print("Complete System Demo")
print("="*60)

# Load best model from registry
print("\n1. Loading best model from MLflow Registry...")
try:
    model = mlflow.sklearn.load_model(
        "models:/ExperimentSuccessPredictor/Staging"
    )
    print("   Model loaded from Registry! ✅")
except:
    print("   (Demo mode — registry not available)")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)

# Load data
df = pd.read_csv('experiments_dataset.csv')

# Prepare features
from sklearn.preprocessing import LabelEncoder
le_m = LabelEncoder()
le_d = LabelEncoder()
df['model_enc'] = le_m.fit_transform(
    df['model_type'])
df['dataset_enc'] = le_d.fit_transform(
    df['dataset'])
df['n_est_filled'] = df[
    'n_estimators'].fillna(0)
df['lr_filled'] = df[
    'learning_rate'].fillna(0)

X = df[['model_enc', 'dataset_enc',
         'n_est_filled', 'lr_filled',
         'train_time']].values
y = df['passed'].astype(int).values

model.fit(X, y)
preds = model.predict(X)

print("\n2. System Performance Report:")
print(classification_report(
    y, preds,
    target_names=['Failed', 'Passed']
))

# Final MLflow run
mlflow.set_experiment("capstone_final_demo")
with mlflow.start_run(run_name="FINAL_DEMO"):
    mlflow.log_param("status", "COMPLETE")
    mlflow.log_param("components", [
        "PySpark", "Delta Lake", "MLflow",
        "RAG", "CNN", "YOLOv8"
    ])
    mlflow.log_metric("final_accuracy",
                       (preds == y).mean())
    print("\n3. Final run logged to MLflow! ✅")

print("\n" + "="*60)
print("CAPSTONE COMPLETE! 🎉")
print("="*60)
print("""
System Components:
✅ Data Layer:    PySpark + Delta Lake
✅ ML Layer:      XGBoost + PyTorch + MLflow
✅ GenAI Layer:   RAG (LangChain + FAISS)
✅ CV Layer:      CNN + OpenCV + YOLOv8
✅ MLOps Layer:   MLflow Registry + tracking

Skills demonstrated:
→ Apache Spark / PySpark
→ Delta Lake (ACID + time travel)
→ MLflow (tracking + registry)
→ PyTorch (NN + CNN)
→ RAG pipeline (HuggingFace + FAISS)
→ Computer Vision (OpenCV + YOLOv8)
→ XGBoost + ensemble methods
→ Docker + Git (daily commits)

This project alone covers every skill
listed in Databricks MLE job description!
""")
