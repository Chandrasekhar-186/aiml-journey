# Phase 3 Day 4 — SVM + Kernel Trick
# Date: May 12, 2026
# The most mathematically beautiful ML model!

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.datasets import (
    make_classification, make_circles,
    make_moons
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report
)
from sklearn.pipeline import Pipeline
import mlflow

print("="*60)
print("SVM + Kernel Trick — Complete Guide")
print("="*60)

"""
SVM — COMPLETE MATH

Core idea: find hyperplane that MAXIMIZES
the margin between two classes.

Hyperplane: w·x + b = 0
  w = normal vector (perpendicular to hyperplane)
  b = bias (shifts hyperplane)

Support Vectors: training points CLOSEST
  to the hyperplane — they define the margin!

Margin = 2 / ||w||
Maximize margin = Minimize ||w||²

OPTIMIZATION PROBLEM:
Minimize:   (1/2)||w||²
Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i

This is a CONVEX QP — guaranteed global min!

SOFT MARGIN (C parameter):
Real data is rarely linearly separable!
Introduce slack variables ξᵢ ≥ 0:

Minimize: (1/2)||w||² + C * Σξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ

C = regularization parameter:
  Large C:  small margin, few violations
            (low bias, high variance)
  Small C:  large margin, more violations
            (high bias, low variance)
  Tune C with cross-validation!

KERNEL TRICK:
Problem: data not linearly separable in
         original feature space!
Solution: map to HIGHER dimensional space
         where it IS separable!

Key insight: SVM only needs dot products!
K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)

Compute kernel directly WITHOUT computing φ!
This is the kernel trick — O(n²) not O(n^d)!

Common kernels:
Linear:  K(x,z) = x·z
Poly:    K(x,z) = (x·z + c)^d
RBF:     K(x,z) = exp(-γ||x-z||²)
Sigmoid: K(x,z) = tanh(αx·z + c)
"""

# 1. Linear SVM — linearly separable
print("\n=== LINEAR SVM ===")
X_linear, y_linear = make_classification(
    n_samples=500, n_features=2,
    n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = \
    train_test_split(X_linear, y_linear,
                     test_size=0.2,
                     random_state=42)

# Scale! (CRITICAL for SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Linear kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_s, y_train)
acc = accuracy_score(
    y_test, svm_linear.predict(X_test_s)
)
print(f"Linear SVM accuracy: {acc:.4f}")
print(f"Support vectors:     "
      f"{svm_linear.n_support_}")
print(f"Total SVs:           "
      f"{sum(svm_linear.n_support_)}")

# 2. RBF Kernel — non-linearly separable
print("\n=== RBF KERNEL (circles dataset) ===")
X_circles, y_circles = make_circles(
    n_samples=500, noise=0.1,
    factor=0.3, random_state=42
)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_circles, y_circles,
    test_size=0.2, random_state=42
)

# Linear fails on circles!
svm_lin = SVC(kernel='linear')
svm_lin.fit(scaler.fit_transform(X_tr), y_tr)
acc_lin = accuracy_score(
    y_te,
    svm_lin.predict(scaler.transform(X_te))
)

# RBF succeeds!
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(scaler.fit_transform(X_tr), y_tr)
acc_rbf = accuracy_score(
    y_te,
    svm_rbf.predict(scaler.transform(X_te))
)

print(f"Linear kernel: {acc_lin:.4f} ← fails!")
print(f"RBF kernel:    {acc_rbf:.4f} ← succeeds!")
print("""
WHY RBF works on circles:
→ Maps to infinite-dimensional feature space
→ In that space, circles ARE linearly separable!
→ Kernel trick: compute it in O(n²) not O(∞)!
""")

# 3. Moons dataset — polynomial kernel
print("\n=== POLYNOMIAL KERNEL (moons) ===")
X_moons, y_moons = make_moons(
    n_samples=500, noise=0.15, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_moons, y_moons,
    test_size=0.2, random_state=42
)

sc = StandardScaler()
svm_poly = SVC(kernel='poly', degree=3, C=5.0)
svm_poly.fit(sc.fit_transform(X_tr), y_tr)
acc_poly = accuracy_score(
    y_te, svm_poly.predict(sc.transform(X_te))
)
print(f"Polynomial (degree=3): {acc_poly:.4f}")

# 4. Hyperparameter tuning
print("\n=== HYPERPARAMETER TUNING ===")
print("""
C parameter:
  Large C (e.g. 100):  narrow margin
                        few support vectors
                        risk of overfit
  Small C (e.g. 0.1):  wide margin
                        more support vectors
                        more robust

γ (gamma) for RBF:
  Large γ:  small radius, tight fit
            each SV influences small region
            risk of overfit
  Small γ:  large radius, smooth boundary
            each SV influences large region
            risk of underfit
  'scale':  1/(n_features * X.var()) ← good default
  'auto':   1/n_features

Rule of thumb:
  Start: C=1, gamma='scale'
  Tune with GridSearchCV or RandomizedSearchCV
""")

# Grid search
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}

X_full, y_full = make_classification(
    n_samples=800, n_features=10,
    random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_full,
    test_size=0.2, random_state=42
)

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=3, scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_tr, y_tr)
best_acc = accuracy_score(
    y_te, grid_search.predict(X_te)
)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV acc: "
      f"{grid_search.best_score_:.4f}")
print(f"Test acc:    {best_acc:.4f}")

# 5. SVM vs other algorithms
print("\n=== SVM vs RF vs LR ===")
print("""
                SVM     RF      LogReg
High-dim data:  ✅       ⚠️      ✅
Non-linear:     ✅(kern) ✅      ❌
Interpretable:  ❌       ✅      ✅
Speed (train):  Slow     Fast    Fast
Speed (predict):Fast     Fast    Fast
Probabilistic:  ❌       ✅      ✅
Large datasets: ❌(O(n²)) ✅     ✅

SVM sweet spot:
→ Text classification (high-dim, sparse)
→ Image classification (medium-sized)
→ When n_features >> n_samples
→ When you need maximum accuracy on
  small-medium datasets

Don't use SVM when:
→ n > 100,000 (too slow!)
→ Need probability outputs
→ Need interpretability
""")

# 6. SVM Interview Questions
print("\n=== INTERVIEW PREP ===")
print("""
Q: What are support vectors?
A: Training points closest to the decision
   boundary. They DEFINE the margin.
   Remove non-SVs → same model! SVs are all
   that matter.

Q: Why scale features for SVM?
A: SVM maximizes margin based on distances.
   Unscaled: large-magnitude features dominate.
   Scaled:   all features contribute equally.
   ALWAYS scale before SVM!

Q: What does C control?
A: Bias-variance tradeoff.
   Large C: low bias, high variance (tight fit)
   Small C: high bias, low variance (wide margin)

Q: What is the kernel trick?
A: Compute dot products in high-dimensional
   space WITHOUT explicit transformation.
   K(x,z) = φ(x)·φ(z) computed directly.
   Enables non-linear SVMs in O(n²) time!

Q: How does SVM handle multiclass?
A: OvR (one-vs-rest): K binary SVMs
   OvO (one-vs-one): K*(K-1)/2 SVMs → slower
   sklearn default: OvO for SVC
""")

# 7. Log to MLflow
mlflow.set_experiment("phase3_svm")
with mlflow.start_run(run_name="SVM_RBF"):
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("C",
        grid_search.best_params_['svm__C'])
    mlflow.log_param("gamma",
        grid_search.best_params_['svm__gamma'])
    mlflow.log_metric("test_accuracy", best_acc)
    mlflow.log_metric(
        "cv_score",
        grid_search.best_score_
    )
    print("\nSVM logged to MLflow!")

print("\n" + "="*60)
print("SVM + Kernel Trick — MASTERED! ⚡")
print("="*60)
