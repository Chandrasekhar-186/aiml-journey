# Phase 3 Day 9 — Optimizers + BatchNorm
# Date: May 17, 2026
# Training tricks that make deep learning work!

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader, TensorDataset
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

print("="*60)
print("Optimizers + BatchNorm + Dropout")
print("="*60)

"""
OPTIMIZERS — WHY SGD ISN'T ENOUGH

Vanilla Gradient Descent:
θ := θ - α * ∇J(θ)

Problems:
1. Same learning rate for ALL parameters
2. Slow convergence in ravines
3. Gets stuck in local minima / saddle points
4. Noisy gradient with mini-batches

Solutions: momentum, adaptive learning rates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SGD WITH MOMENTUM:
v_t := β * v_{t-1} + (1-β) * ∇J(θ)
θ   := θ - α * v_t

Physical analogy: ball rolling down hill
  → builds speed (momentum) in consistent dir
  → dampens oscillations
β = 0.9 typical

NESTEROV MOMENTUM:
"Look ahead" before computing gradient
θ_ahead = θ - α * β * v
v_t := β * v_{t-1} + ∇J(θ_ahead)
θ   := θ - α * v_t
→ Usually better than standard momentum

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADAGRAD:
Adapts learning rate PER PARAMETER!
G_t := G_{t-1} + (∇J)²
θ   := θ - (α / √(G_t + ε)) * ∇J

→ Frequent params: small lr
→ Rare params:    large lr
→ Problem: G grows forever → lr → 0
           (training stops too early!)

RMSPROP (fixes AdaGrad):
G_t := ρ * G_{t-1} + (1-ρ) * (∇J)²
θ   := θ - (α / √(G_t + ε)) * ∇J

→ Exponential moving average of squared grad
→ Doesn't accumulate forever!
→ ρ = 0.9 typical

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADAM (Adaptive Moment Estimation):
The default choice for most deep learning!

Combines: Momentum + RMSProp

m_t := β₁ * m_{t-1} + (1-β₁) * ∇J
v_t := β₂ * v_{t-1} + (1-β₂) * (∇J)²

Bias correction (crucial for early steps!):
m̂_t := m_t / (1 - β₁^t)
v̂_t := v_t / (1 - β₂^t)

Update:
θ := θ - α * m̂_t / (√v̂_t + ε)

Default hyperparameters:
α = 0.001 (learning rate)
β₁ = 0.9  (first moment — momentum)
β₂ = 0.999 (second moment — RMSProp)
ε = 1e-8   (numerical stability)

WHY ADAM WORKS SO WELL:
→ Adapts lr per parameter (like RMSProp)
→ Uses momentum for direction (like SGD+mom)
→ Bias correction → good from step 1
→ Invariant to gradient scale
"""

# 1. Implement optimizers from scratch
class SGD_Scratch:
    def __init__(self, lr=0.01,
                  momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.v:
                self.v[key] = np.zeros_like(
                    params[key]
                )
            self.v[key] = (
                self.momentum * self.v[key] -
                self.lr * grads[f'd{key}']
            )
            params[key] += self.v[key]

class Adam_Scratch:
    def __init__(self, lr=0.001,
                  beta1=0.9, beta2=0.999,
                  eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(
                    params[key]
                )
                self.v[key] = np.zeros_like(
                    params[key]
                )
            g = grads[f'd{key}']

            # Moment updates
            self.m[key] = (
                self.beta1 * self.m[key] +
                (1 - self.beta1) * g
            )
            self.v[key] = (
                self.beta2 * self.v[key] +
                (1 - self.beta2) * g**2
            )

            # Bias correction
            m_hat = self.m[key] / \
                (1 - self.beta1**self.t)
            v_hat = self.v[key] / \
                (1 - self.beta2**self.t)

            # Update
            params[key] -= (
                self.lr * m_hat /
                (np.sqrt(v_hat) + self.eps)
            )

print("Optimizers implemented from scratch!")
print("Adam = Momentum + RMSProp + Bias correction")

# 2. PyTorch comparison
print("\n=== PYTORCH OPTIMIZER COMPARISON ===")
X_data, y_data = make_classification(
    n_samples=2000, n_features=20,
    n_informative=12, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_data,
    test_size=0.2, random_state=42
)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

# Convert to tensors
X_train_t = torch.FloatTensor(X_tr_s)
y_train_t = torch.FloatTensor(y_tr)
X_test_t = torch.FloatTensor(X_te_s)
y_test_t = torch.FloatTensor(y_te)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(
    train_ds, batch_size=64, shuffle=True
)

def build_model():
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

def train_model(model, optimizer,
                 epochs=100):
    criterion = nn.BCELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
    return losses

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = (model(X).squeeze() >= 0.5)\
            .float()
        return (pred == y).float().mean().item()

# Compare optimizers
optimizers_config = {
    'SGD': lambda p: optim.SGD(p, lr=0.01),
    'SGD+Momentum': lambda p: optim.SGD(
        p, lr=0.01, momentum=0.9),
    'RMSProp': lambda p: optim.RMSprop(
        p, lr=0.001),
    'Adam': lambda p: optim.Adam(p, lr=0.001),
    'AdamW': lambda p: optim.AdamW(
        p, lr=0.001, weight_decay=0.01),
}

results = {}
mlflow.set_experiment("phase3_optimizers")

with mlflow.start_run(
        run_name="optimizer_comparison"):
    for name, opt_fn in optimizers_config.items():
        torch.manual_seed(42)
        model = build_model()
        optimizer = opt_fn(model.parameters())
        losses = train_model(
            model, optimizer, epochs=100
        )
        acc = evaluate(model, X_test_t,
                        y_test_t)
        results[name] = acc
        mlflow.log_metric(f"{name}_acc", acc)
        mlflow.log_metric(
            f"{name}_final_loss", losses[-1]
        )
        print(f"{name:15}: acc={acc:.4f}, "
              f"loss={losses[-1]:.4f}")

best = max(results, key=results.get)
print(f"\nBest optimizer: {best} "
      f"({results[best]:.4f})")
mlflow.log_param("best_optimizer", best)

# 3. Batch Normalization
print("\n=== BATCH NORMALIZATION ===")
print("""
Problem: Internal Covariate Shift
→ As weights update, distribution of each
  layer's inputs shifts → slow learning!

BatchNorm solution:
For each mini-batch:
1. μ_B = (1/m) Σ xᵢ        (batch mean)
2. σ²_B = (1/m) Σ(xᵢ-μ_B)² (batch variance)
3. x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)  (normalize)
4. yᵢ = γ * x̂ᵢ + β         (scale + shift)
   γ, β = LEARNABLE parameters!

Benefits:
→ Stabilizes training (higher lr possible!)
→ Reduces dependence on initialization
→ Acts as regularizer (slight noise per batch)
→ Enables deeper networks
→ Faster convergence

Placement: BEFORE activation (common)
           or AFTER activation (both work)

At inference: use running mean/variance
              (computed during training)

Layer Normalization (for Transformers!):
→ Normalize across FEATURES (not batch)
→ Works for any batch size (even 1!)
→ Better for sequential data (NLP)
""")

# BatchNorm in PyTorch
class ModelWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),  # ← after linear!
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
model_bn = ModelWithBN()
optimizer_bn = optim.Adam(
    model_bn.parameters(), lr=0.001
)
losses_bn = train_model(
    model_bn, optimizer_bn, epochs=100
)
acc_bn = evaluate(model_bn, X_test_t,
                   y_test_t)
print(f"With BatchNorm + Adam: {acc_bn:.4f}")
print(f"Without BatchNorm:     "
      f"{results['Adam']:.4f}")

# 4. Dropout
print("\n=== DROPOUT REGULARIZATION ===")
print("""
Dropout: randomly zero out neurons during training!

Forward pass (training):
  Each neuron kept with probability p (keep_prob)
  Zeroed with probability (1-p)
  Scale remaining: divide by p (inverted dropout)

Forward pass (inference):
  ALL neurons active (no dropout)
  No scaling needed (inverted handles it!)

WHY it works:
→ Forces redundant representations
→ Each neuron can't rely on specific others
→ Effectively trains ensemble of 2^n networks
→ Strong regularizer (prevents overfitting)

Where to apply:
→ After large fully-connected layers
→ NOT on batch norm layers usually
→ NOT on output layer
→ Rate: 0.2-0.5 typical

For CNN: use SpatialDropout (whole channels)
For RNN: use VariationalDropout (same mask)
""")

class ModelWithDropout(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),   # ← dropout!
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
model_do = ModelWithDropout(dropout=0.3)
optimizer_do = optim.Adam(
    model_do.parameters(), lr=0.001
)
losses_do = train_model(
    model_do, optimizer_do, epochs=100
)
acc_do = evaluate(model_do, X_test_t,
                   y_test_t)
print(f"BatchNorm + Dropout + Adam: {acc_do:.4f}")

mlflow.set_experiment("phase3_regularization")
with mlflow.start_run(
        run_name="batchnorm_dropout"):
    mlflow.log_metric("acc_baseline",
                       results['Adam'])
    mlflow.log_metric("acc_batchnorm", acc_bn)
    mlflow.log_metric("acc_dropout", acc_do)
    mlflow.log_param("dropout_rate", 0.3)
    print("\nRegularization logged!")

print("\n" + "="*60)
print("Optimizers + BatchNorm + Dropout — DONE! ⚡")
print("="*60)
