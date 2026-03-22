# Day 08 — MLflow Introduction
# Date: March 20, 2026

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment name
mlflow.set_experiment("iris_classification")

# Start MLflow run — logs EVERYTHING!
with mlflow.start_run():
    # Define parameters
    n_estimators = 100
    max_depth = 5

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset", "iris")

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("n_test_samples", len(X_test))

    # Log model
    mlflow.sklearn.log_model(model, "random_forest")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run logged to MLflow!")

# View results
print("\nTo view MLflow UI, run:")
print("mlflow ui")
print("Then open: http://localhost:5000")
