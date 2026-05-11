# Phase 3 Day 1 — Linear Regression Math
# Date: May 9, 2026
# Goal: Understand from FIRST PRINCIPLES!

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import mlflow

print("="*60)
print("Linear Regression — First Principles")
print("="*60)

"""
LINEAR REGRESSION — COMPLETE MATH

Hypothesis:
h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
       = θᵀx  (vector form, x₀=1)

Cost Function (Mean Squared Error):
J(θ) = (1/2m) Σᵢ (h_θ(xⁱ) - yⁱ)²

WHY (1/2m)?
→ m: average over training examples
→ 1/2: cancels with derivative (cleanup)
→ Minimizing J(θ) = best fit line

Gradient Descent:
Repeat until convergence:
    θ_j := θ_j - α * ∂J(θ)/∂θ_j

Partial derivative:
∂J(θ)/∂θ_j = (1/m) Σᵢ (h_θ(xⁱ) - yⁱ) * xⱼⁱ

In matrix form:
θ := θ - (α/m) * Xᵀ(Xθ - y)

Normal Equation (closed-form solution):
θ = (XᵀX)⁻¹Xᵀy

When to use Normal Equation vs GD:
Normal: n < 10,000 features (matrix inversion!)
GD:     n >= 10,000 features (scales better)
"""

# 1. Implement Linear Regression from SCRATCH
class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.theta = None
        self.costs = []

    def fit(self, X, y):
        m, n = X.shape
        # Add bias term (x₀ = 1)
        X_b = np.c_[np.ones((m, 1)), X]
        # Initialize θ to zeros
        self.theta = np.zeros(n + 1)

        for epoch in range(self.epochs):
            # Predictions
            predictions = X_b @ self.theta
            # Errors
            errors = predictions - y
            # Gradient
            gradient = (1/m) * X_b.T @ errors
            # Update
            self.theta -= self.lr * gradient
            # Cost
            cost = (1/(2*m)) * np.sum(errors**2)
            self.costs.append(cost)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: "
                      f"cost={cost:.6f}")

        return self

    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b @ self.theta

    def normal_equation(self, X, y):
        """Closed-form solution"""
        X_b = np.c_[np.ones((len(X), 1)), X]
        # θ = (XᵀX)⁻¹Xᵀy
        self.theta = np.linalg.pinv(
            X_b.T @ X_b
        ) @ X_b.T @ y
        return self

# 2. Generate synthetic data
np.random.seed(42)
m = 200
X = 2 * np.random.randn(m, 1)
y = 4 + 3 * X.squeeze() + \
    np.random.randn(m) * 0.5

print("\n=== GRADIENT DESCENT TRAINING ===")
model_gd = LinearRegressionScratch(
    lr=0.1, epochs=500
)
model_gd.fit(X, y)
print(f"\nLearned θ (GD):     {model_gd.theta}")
print(f"True parameters:    [4.0, 3.0]")

print("\n=== NORMAL EQUATION ===")
model_ne = LinearRegressionScratch()
model_ne.normal_equation(X, y)
print(f"Learned θ (Normal): {model_ne.theta}")

# 3. Compare with sklearn
sk_model = LinearRegression()
sk_model.fit(X, y)
print(f"sklearn intercept:  {sk_model.intercept_:.4f}")
print(f"sklearn coef:       {sk_model.coef_[0]:.4f}")

# 4. Evaluation metrics
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error
)
y_pred = model_gd.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n=== EVALUATION METRICS ===")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}  ← in same units as y!")
print(f"MAE:  {mae:.4f}  ← robust to outliers")
print(f"R²:   {r2:.4f}  ← 1.0 = perfect fit")

# 5. Regularization
print("\n=== REGULARIZATION ===")
print("""
Why regularization?
Without: model overfits training data
With:    penalizes large θ → simpler model

Ridge (L2) regularization:
J(θ) = MSE + λ * Σθⱼ²
→ Shrinks all features proportionally
→ Never exactly zero (keeps all features)
→ Best when: all features somewhat useful

Lasso (L1) regularization:
J(θ) = MSE + λ * Σ|θⱼ|
→ Can shrink features to EXACTLY zero
→ Automatic feature selection!
→ Best when: many irrelevant features

Elastic Net (L1 + L2):
J(θ) = MSE + λ₁*Σ|θⱼ| + λ₂*Σθⱼ²
→ Best of both worlds
→ Best when: many features, some correlated

λ (lambda) = regularization strength:
λ = 0:   no regularization (overfit risk)
λ = ∞:   all θ → 0 (underfit)
λ = ???: tune with cross-validation!
""")

# Ridge from scratch
class RidgeRegression:
    def __init__(self, lr=0.01,
                  epochs=1000, lam=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)

        for _ in range(self.epochs):
            predictions = X_b @ self.theta
            errors = predictions - y
            gradient = (1/m) * X_b.T @ errors
            # L2 penalty (don't regularize bias!)
            reg = self.lam * self.theta.copy()
            reg[0] = 0  # don't penalize θ₀
            self.theta -= self.lr * (
                gradient + reg
            )
        return self

ridge = RidgeRegression(
    lr=0.1, epochs=500, lam=0.5
)
ridge.fit(X, y)
print(f"Ridge θ (λ=0.5): {ridge.theta}")
print(f"GD θ   (λ=0.0):  {model_gd.theta}")
print("Ridge shrinks θ toward 0 ✅")

# 6. Feature scaling importance
print("\n=== FEATURE SCALING ===")
print("""
WHY scaling matters for gradient descent:
Unscaled: θ₁ and θ₂ on different scales →
          gradient descent oscillates!
Scaled:   equal contribution → fast convergence

StandardScaler: z = (x - μ) / σ
MinMaxScaler:   x' = (x - min) / (max - min)

Rule: ALWAYS scale for GD-based algorithms!
Not needed for: Normal equation, tree models
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_scaled = LinearRegressionScratch(
    lr=0.1, epochs=500
)
model_scaled.fit(X_scaled, y)
print(f"Epochs with scaling:    "
      f"{len(model_scaled.costs)}")
print(f"Final cost with scaling: "
      f"{model_scaled.costs[-1]:.6f}")

# 7. Log everything to MLflow
mlflow.set_experiment("phase3_linear_regression")
with mlflow.start_run(
        run_name="linear_reg_from_scratch"):
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("epochs", 500)
    mlflow.log_param("regularization", "none")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("final_theta_0",
                       model_gd.theta[0])
    mlflow.log_metric("final_theta_1",
                       model_gd.theta[1])

    # Log learning curve
    plt.figure(figsize=(8, 4))
    plt.plot(model_gd.costs)
    plt.xlabel("Epoch")
    plt.ylabel("Cost J(θ)")
    plt.title("Learning Curve — GD Convergence")
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    mlflow.log_artifact("learning_curve.png")
    print("\nLinear regression logged to MLflow!")

print("\n" + "="*60)
print("PHASE 3 DAY 1 COMPLETE! 🚀")
print("Linear Regression mastered from scratch!")
print("="*60)
