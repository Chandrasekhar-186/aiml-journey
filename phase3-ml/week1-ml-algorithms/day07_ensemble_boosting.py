# Phase 3 Day 7 — Ensemble Methods + XGBoost
# Date: May 15, 2026
# Most powerful tabular ML algorithm!

import numpy as np
from sklearn.datasets import (
    make_classification, make_regression
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
import mlflow

print("="*60)
print("Ensemble Methods + XGBoost")
print("="*60)

"""
ENSEMBLE METHODS — COMPLETE MATH

Three pillars of ensemble learning:
1. Bagging:  train in PARALLEL on subsets
             → reduces VARIANCE
             → example: Random Forest

2. Boosting: train SEQUENTIALLY, each model
             corrects previous errors
             → reduces BIAS
             → example: XGBoost, AdaBoost

3. Stacking: train meta-model on predictions
             of base models
             → reduces both bias + variance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADABOOST (Adaptive Boosting):

Algorithm:
1. Initialize equal weights: wᵢ = 1/m
2. For t = 1..T:
   a. Train weak learner hₜ on weighted data
   b. Compute error: εₜ = Σwᵢ * 1[hₜ(xᵢ)≠yᵢ]
   c. Compute learner weight:
      αₜ = (1/2) * ln((1-εₜ)/εₜ)
   d. Update sample weights:
      wᵢ ← wᵢ * exp(-αₜ * yᵢ * hₜ(xᵢ))
      (increase weight for misclassified!)
3. Final: H(x) = sign(Σ αₜ * hₜ(x))

Key insight: misclassified samples get
             HIGHER weight → next model
             focuses on hard examples!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRADIENT BOOSTING:

More general than AdaBoost!
Works with ANY differentiable loss function.

Algorithm:
1. Initialize: F₀(x) = argmin_γ Σ L(yᵢ, γ)
   (e.g. mean for regression, log-odds for class)
2. For t = 1..T:
   a. Compute pseudo-residuals:
      rᵢₜ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
      (negative gradient of loss w.r.t. prediction!)
   b. Fit tree hₜ to residuals rᵢₜ
   c. Find step size γₜ (line search)
   d. Update: Fₜ(x) = Fₜ₋₁(x) + ν*γₜ*hₜ(x)
      ν = learning rate (shrinkage)

Key insight: fit each tree to the GRADIENT
             of the loss (residual errors)!
             → Gradient descent in function space!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

XGBoost — EXTREME GRADIENT BOOSTING:

Improvements over vanilla GBM:
1. Regularization: L1+L2 in objective!
   Obj = Σ L(yᵢ,ŷᵢ) + Σ Ω(fₜ)
   Ω(f) = γT + (1/2)λ||w||²
   (T = #leaves, w = leaf weights)

2. Second-order optimization:
   Uses both gradient (gᵢ) AND hessian (hᵢ)!
   Better convergence than first-order only.

3. Column subsampling:
   Like Random Forest feature subsampling
   → Less overfitting, faster training

4. Parallel tree construction:
   Sort features in parallel → fast!
   (unlike sklearn GBM which is sequential)

5. Handling missing values:
   Learns optimal direction for missing data!
"""

# 1. AdaBoost
print("\n=== ADABOOST ===")
X, y = make_classification(
    n_samples=2000, n_features=20,
    n_informative=12, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1  # "stump"!
    ),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_tr, y_tr)
ada_acc = accuracy_score(y_te,
                          ada.predict(X_te))
print(f"AdaBoost accuracy: {ada_acc:.4f}")

# 2. Gradient Boosting
print("\n=== GRADIENT BOOSTING ===")
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,     # stochastic GBM!
    max_features='sqrt',
    random_state=42
)
gbm.fit(X_tr, y_tr)
gbm_acc = accuracy_score(y_te,
                           gbm.predict(X_te))
print(f"GBM accuracy: {gbm_acc:.4f}")

# Feature importance from GBM
importances = gbm.feature_importances_
top_5 = np.argsort(importances)[::-1][:5]
print("Top 5 features (GBM):")
for rank, f in enumerate(top_5, 1):
    print(f"  {rank}. Feature {f:2d}: "
          f"{importances[f]:.4f}")

# 3. XGBoost — the king of tabular ML!
print("\n=== XGBOOST ===")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,  # feature subsampling
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    verbose=False
)
xgb_acc = accuracy_score(
    y_te, xgb_model.predict(X_te)
)
print(f"XGBoost accuracy: {xgb_acc:.4f}")

# 4. Stacking
print("\n=== STACKING ===")
base_models = [
    ('rf', GradientBoostingClassifier(
        n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf',
                 probability=True,
                 random_state=42)),
    ('lr', LogisticRegression(
        max_iter=500, random_state=42))
]
meta_model = LogisticRegression(
    max_iter=500, random_state=42
)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,           # cross-val for meta-features
    n_jobs=-1
)
stacking.fit(X_tr, y_tr)
stack_acc = accuracy_score(
    y_te, stacking.predict(X_te)
)
print(f"Stacking accuracy: {stack_acc:.4f}")

# 5. Comparison
print("\n=== FULL COMPARISON ===")
print(f"{'Algorithm':20} {'Accuracy':>10}")
print("-" * 32)
results = {
    'AdaBoost': ada_acc,
    'GradientBoosting': gbm_acc,
    'XGBoost': xgb_acc,
    'Stacking': stack_acc,
}
for name, acc in sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True):
    bar = "█" * int(acc * 30)
    print(f"{name:20} {acc:.4f} {bar}")

# 6. XGBoost hyperparameter guide
print("\n=== XGBOOST TUNING GUIDE ===")
print("""
Most important parameters:

n_estimators:     number of trees (100-1000)
learning_rate:    shrinkage (0.01-0.3)
                  lower = better but slower
max_depth:        tree depth (3-10)
                  lower = less overfit
subsample:        row sampling (0.6-1.0)
colsample_bytree: column sampling (0.6-1.0)
reg_alpha:        L1 regularization
reg_lambda:       L2 regularization

Golden starting point:
  n_estimators=500, learning_rate=0.05,
  max_depth=6, subsample=0.8,
  colsample_bytree=0.8

Early stopping (crucial!):
  eval_set=[(X_val, y_val)]
  early_stopping_rounds=50
  → Stops when val score doesn't improve
  → Prevents overfitting automatically!

Tuning strategy:
1. Fix learning_rate=0.1, tune tree params
2. Tune subsample + colsample
3. Add regularization (alpha, lambda)
4. Lower learning_rate + increase n_estimators
""")

# 7. Log all to MLflow
mlflow.set_experiment("phase3_ensemble")
with mlflow.start_run(run_name="ensemble_comparison"):
    for name, acc in results.items():
        mlflow.log_metric(
            f"{name.lower()}_acc", acc
        )
    mlflow.log_metric(
        "best_acc", max(results.values())
    )
    mlflow.log_param(
        "best_model",
        max(results, key=results.get)
    )
    print("\nEnsemble comparison logged!")

print("\n" + "="*60)
print("Ensemble + XGBoost — MASTERED! 🚀")
print("Week 1 Phase 3 — COMPLETE! 🏆")
print("="*60)
