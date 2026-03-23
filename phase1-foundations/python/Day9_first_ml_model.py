# Day 09 — First Complete ML Pipeline
# Date: March 21, 2026
# Stack: Scikit-learn + MLflow + Pandas

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                              precision_score,
                              recall_score,
                              f1_score,
                              confusion_matrix)

# 1. Load & explore data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{pd.Series(y).value_counts()}")
print(f"Features: {list(X.columns[:5])}...")

# 2. Feature engineering
X['mean_radius_squared'] = X['mean radius'] ** 2
X['radius_texture_ratio'] = (X['mean radius'] /
                               X['mean texture'])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train multiple models — log ALL to MLflow!
mlflow.set_experiment("breast_cancer_classification")

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42)
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log model name
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features_engineered", 2)

        # Train
        if model_name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log ALL metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Cross validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Log model artifact
        mlflow.sklearn.log_model(model, model_name)

        print(f"\n{model_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV Mean:   {cv_scores.mean():.4f}")

print("\nAll runs logged! Run 'mlflow ui' to compare.")
