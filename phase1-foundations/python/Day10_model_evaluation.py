# Day 10 — Model Evaluation Mastery
# Date: March 22, 2026

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import (train_test_split,
                                      GridSearchCV,
                                      learning_curve)
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              roc_auc_score,
                              roc_curve)
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("model_evaluation_mastery")

with mlflow.start_run(run_name="RF_with_GridSearch"):

    # 1. Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5,
        scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best params
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(param, value)

    # 2. Full evaluation report
    preds = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, preds))

    # 3. ROC AUC — critical metric!
    roc_auc = roc_auc_score(y_test, proba)
    mlflow.log_metric("roc_auc", roc_auc)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric("true_positives", int(tp))
    mlflow.log_metric("false_positives", int(fp))
    mlflow.log_metric("false_negatives", int(fn))

    # 5. Feature importance
    importances = best_model.feature_importances_
    top_features = np.argsort(importances)[-5:]
    print(f"\nTop 5 features:")
    for idx in top_features:
        feat_name = load_breast_cancer().feature_names[idx]
        print(f"  {feat_name}: {importances[idx]:.4f}")

    # 6. Save ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr,
             label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    mlflow.log_artifact('roc_curve.png')
    print("ROC curve saved + logged to MLflow!")

    mlflow.sklearn.log_model(best_model, "best_rf_model")
    print(f"\nBest params: {grid_search.best_params_}")
