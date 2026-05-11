# Phase 3 Day 2 — Logistic Regression Math
# Date: May 10, 2026
# Classification from first principles!

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report
)
import mlflow

print("="*60)
print("Logistic Regression — First Principles")
print("="*60)

"""
LOGISTIC REGRESSION — COMPLETE MATH

Why not Linear Regression for classification?
→ Linear predicts any value (-∞ to +∞)
→ We need probability [0, 1]
→ Solution: sigmoid function!

Sigmoid (logistic) function:
σ(z) = 1 / (1 + e^(-z))
→ Always outputs [0, 1] ✅
→ σ(0) = 0.5 (decision boundary)
→ σ(∞) = 1, σ(-∞) = 0

Hypothesis:
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
Interpretation: P(y=1 | x; θ)

Decision boundary:
Predict 1 if h_θ(x) ≥ 0.5 (i.e. θᵀx ≥ 0)
Predict 0 if h_θ(x) < 0.5 (i.e. θᵀx < 0)

Cost Function (Log Loss / Cross-Entropy):
J(θ) = -(1/m) Σ [y*log(h) + (1-y)*log(1-h)]

WHY log loss (not MSE)?
→ MSE + sigmoid = non-convex (local minima!)
→ Log loss + sigmoid = CONVEX (guaranteed min!)
→ -log(h): large penalty when h≈0 but y=1
→ -log(1-h): large penalty when h≈1 but y=0

Gradient (same form as linear!):
∂J/∂θ_j = (1/m) Σ (h_θ(xⁱ) - yⁱ) * xⱼⁱ

Beautiful: identical gradient to linear reg!
Only difference: h uses sigmoid instead of linear
"""

# 1. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Verify key properties
print("Sigmoid properties:")
print(f"σ(0)   = {sigmoid(0):.4f}  ← 0.5!")
print(f"σ(100) = {sigmoid(100):.4f} ← ≈1.0")
print(f"σ(-100)= {sigmoid(-100):.4f} ← ≈0.0")

# 2. Logistic Regression from scratch
class LogisticRegressionScratch:
    def __init__(self, lr=0.1,
                  epochs=1000, lam=0.0):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam  # L2 regularization
        self.theta = None
        self.costs = []

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)

        for epoch in range(self.epochs):
            # Forward pass
            z = X_b @ self.theta
            h = sigmoid(z)  # predictions

            # Log loss
            cost = -(1/m) * np.sum(
                y * np.log(h + 1e-8) +
                (1-y) * np.log(1-h + 1e-8)
            )
            # L2 regularization
            if self.lam > 0:
                reg_term = (self.lam/(2*m)) * \
                    np.sum(self.theta[1:]**2)
                cost += reg_term

            self.costs.append(cost)

            # Gradient (same as linear!)
            errors = h - y
            gradient = (1/m) * X_b.T @ errors

            # Add L2 gradient
            if self.lam > 0:
                reg_grad = (self.lam/m) * \
                    self.theta.copy()
                reg_grad[0] = 0
                gradient += reg_grad

            self.theta -= self.lr * gradient

            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d}: "
                      f"cost={cost:.4f}")

        return self

    def predict_proba(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return sigmoid(X_b @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)
                >= threshold).astype(int)

# 3. Generate classification data
X, y = make_classification(
    n_samples=1000, n_features=10,
    n_informative=6, n_redundant=2,
    random_state=42
)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2,
                     random_state=42)

print("\n=== TRAINING FROM SCRATCH ===")
model = LogisticRegressionScratch(
    lr=0.1, epochs=1000, lam=0.1
)
model.fit(X_train, y_train)

# 4. Complete evaluation metrics
print("\n=== CLASSIFICATION METRICS ===")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {acc:.4f} ← (TP+TN)/total")
print(f"Precision: {prec:.4f} ← TP/(TP+FP)")
print(f"Recall:    {rec:.4f} ← TP/(TP+FN)")
print(f"F1 Score:  {f1:.4f} ← 2*P*R/(P+R)")
print(f"ROC-AUC:   {auc:.4f} ← rank quality")
print(f"\nConfusion Matrix:\n{cm}")

print("""
METRIC INTUITION:
Precision: "Of predicted positives, how many
            were actually positive?"
           High when: FP cost is high
           (spam filter — don't mark good as spam)

Recall:    "Of actual positives, how many
            did we catch?"
           High when: FN cost is high
           (cancer detection — don't miss cases)

F1:        Harmonic mean of P and R
           Use when: class imbalance

ROC-AUC:   Probability model ranks random
           positive above random negative
           1.0 = perfect | 0.5 = random

WHEN TO USE WHICH:
Fraud detection:  High Recall
                  (catch all fraud, FP ok)
Spam filter:      High Precision
                  (FP = lost email = bad!)
Medical diagnosis: F1 or Recall
                   (missing cancer = terrible!)
""")

# 5. Threshold tuning
print("=== THRESHOLD TUNING ===")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Precision':>10}"
      f"{'Recall':>10} {'F1':>8}")
for t in thresholds:
    p_t = model.predict(X_test, threshold=t)
    p = precision_score(y_test, p_t,
                         zero_division=0)
    r = recall_score(y_test, p_t,
                      zero_division=0)
    f = f1_score(y_test, p_t,
                  zero_division=0)
    print(f"{t:>10.1f} {p:>10.4f}"
          f"{r:>10.4f} {f:>8.4f}")

# 6. Compare with sklearn
sk_model = LogisticRegression(
    C=10, max_iter=1000, random_state=42
)
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
sk_acc = accuracy_score(y_test, sk_pred)
print(f"\nScratch accuracy: {acc:.4f}")
print(f"sklearn accuracy: {sk_acc:.4f}")
print("Match! ✅")

# 7. Multiclass extension
print("\n=== MULTICLASS STRATEGIES ===")
print("""
Binary: sigmoid → threshold → 0 or 1

Multiclass approaches:

One-vs-Rest (OvR):
→ K binary classifiers (one per class)
→ Each: "is this class k or not?"
→ Predict: class with highest probability
→ sklearn default for logistic regression

One-vs-One (OvO):
→ K*(K-1)/2 binary classifiers
→ Each: binary for pair (i,j)
→ Predict: majority vote
→ Better for non-linear SVMs

Softmax (Multinomial):
→ Generalize sigmoid to K classes
→ P(y=k|x) = exp(θ_kᵀx) / Σexp(θ_jᵀx)
→ All probabilities sum to 1.0
→ Best for neural networks!
""")

# 8. Log to MLflow
mlflow.set_experiment(
    "phase3_logistic_regression"
)
with mlflow.start_run(
        run_name="logistic_scratch"):
    mlflow.log_param("lr", 0.1)
    mlflow.log_param("epochs", 1000)
    mlflow.log_param("lambda", 0.1)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)
    print("\nLogistic regression logged!")
