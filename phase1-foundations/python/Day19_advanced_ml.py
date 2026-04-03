# Day 19 — Advanced ML Algorithms
# Date: March 31, 2026
# Gradient Boosting + XGBoost + Comparison

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_california_housing
from sklearn.model_selection import (
    train_test_split, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import xgboost as xgb

# 1. Load regression dataset
X, y = load_california_housing(return_X_y=True)
feature_names = load_california_housing().feature_names
X = pd.DataFrame(X, columns=feature_names)

print(f"Dataset: {X.shape}")
print(f"Target: House prices (median)")
print(f"Features: {list(X.columns)}")

# 2. Feature engineering
X['rooms_per_person'] = (X['AveRooms'] /
                          X['AveOccup'])
X['bedrooms_ratio'] = (X['AveBedrms'] /
                        X['AveRooms'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Compare all ensemble methods
mlflow.set_experiment("ensemble_comparison")

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100, random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=42,
        verbosity=0
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=100, learning_rate=0.1,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Log params
        mlflow.log_param("model", name)
        mlflow.log_param("n_estimators", 100)

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(
            y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Cross validation
        cv = cross_val_score(
            model, X_train, y_train,
            cv=3, scoring='r2', n_jobs=-1
        )

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("cv_r2_mean",
                           cv.mean())
        mlflow.sklearn.log_model(model, name)

        results[name] = {
            "rmse": rmse, "mae": mae,
            "r2": r2, "cv_r2": cv.mean()
        }
        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  CV R²: {cv.mean():.4f}")

# 4. Algorithm comparison notes
print("\n" + "="*55)
print("ALGORITHM COMPARISON — Interview Ready!")
print("="*55)
print("""
Random Forest:
  + Parallel training (fast!)
  + Low variance, handles overfitting well
  + Feature importance built-in
  - Higher memory usage
  - Slower prediction than boosting

Gradient Boosting:
  + Usually better accuracy than RF
  + Handles mixed data types well
  - Sequential training (slower)
  - More hyperparameters to tune
  - Prone to overfitting if not tuned

XGBoost:
  + Regularization (L1 + L2) built-in
  + Handles missing values automatically
  + GPU support
  + Usually best performance
  - More complex to tune

AdaBoost:
  + Simple, interpretable
  + Works with weak learners
  - Sensitive to noisy data
  - Slower than modern methods

When to use what:
  Tabular data + accuracy: XGBoost
  Need interpretability:   Random Forest
  Large dataset + speed:   Random Forest
  Small dataset:           Gradient Boosting
""")

# 5. Feature importance
print("Top 5 features (XGBoost):")
xgb_model = models["XGBoost"]
importances = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
for feat, imp in importances.head(5).items():
    print(f"  {feat:20s}: {imp:.4f}")
