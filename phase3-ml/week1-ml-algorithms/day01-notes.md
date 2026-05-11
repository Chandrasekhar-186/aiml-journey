# Phase 3 Day 1 Notes — Linear Regression

## Core Formula Chain
Data → h_θ(x) = θᵀx → J(θ) = MSE →
∂J/∂θ → θ update → convergence!

## Cost Function Intuition
J(θ) = (1/2m) Σ(predicted - actual)²
→ Large error → large J
→ Gradient descent minimizes J
→ At minimum: best possible θ!

## Gradient Descent Key Rules
α too large:  overshoots → diverges!
α too small:  converges too slow
α just right: smooth convergence
Batch GD:     uses ALL examples per step
SGD:          uses ONE example per step
Mini-batch:   uses BATCH_SIZE examples ← best!

## Normal Equation When to Use
Use:     n < 10,000 features
Avoid:   n >= 10,000 (XᵀX inversion = O(n³)!)
Prefer:  GD for large n, neural networks

## Regularization Summary
Ridge (L2): penalize Σθ²  → shrink, never zero
Lasso (L1): penalize Σ|θ| → sparse (feature select!)
ElasticNet: both combined  → best of both

λ tuning: always use cross-validation!

## R² Score Interpretation
R² = 1 - SS_res/SS_tot
R² = 1.0: perfect prediction
R² = 0.0: predicts mean (useless model)
R² < 0.0: worse than mean (terrible!)

## MLlib Connection
spark.ml.LinearRegression uses:
→ Mini-batch gradient descent
→ Distributed: each executor = local gradient
→ Driver: aggregate + update θ
→ Same math, distributed execution!
