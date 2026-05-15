# Week 1 ML Benchmark — Complete Comparison
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
import xgboost as xgb
import mlflow
import time
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Dataset: {X.shape[0]} samples, "
      f"{X.shape[1]} features")

# Define all pipelines
models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000, random_state=42))
    ]),
    'DecisionTree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(
            max_depth=5, random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100, random_state=42,
            n_jobs=-1))
    ]),
    'SVM_RBF': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf',
                     probability=True,
                     random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=100, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42, n_jobs=-1))
    ]),
}

# Run benchmark
cv = StratifiedKFold(n_splits=5,
                      shuffle=True,
                      random_state=42)
results = {}

mlflow.set_experiment("phase3_week1_benchmark")
with mlflow.start_run(run_name="week1_benchmark"):
    for name, pipeline in models.items():
        start = time.time()
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv, scoring='roc_auc',
            n_jobs=-1
        )
        elapsed = time.time() - start
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'time': elapsed
        }
        mlflow.log_metric(
            f"{name}_auc", scores.mean()
        )
        print(f"{name:20}: "
              f"AUC={scores.mean():.4f}"
              f" ±{scores.std():.4f}"
              f" ({elapsed:.1f}s)")

    best = max(results,
               key=lambda k: results[k]['mean'])
    mlflow.log_param("best_model", best)
    print(f"\nBest model: {best} "
          f"(AUC={results[best]['mean']:.4f})")
