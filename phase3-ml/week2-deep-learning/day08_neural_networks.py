# Phase 3 Day 8 — Neural Networks Math
# Date: May 16, 2026
# Foundation of ALL deep learning!

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mlflow

print("="*60)
print("Neural Networks — First Principles")
print("="*60)

"""
NEURAL NETWORK — COMPLETE MATH

A neural network = stacked logistic regression!
Each layer learns increasingly abstract features.

ARCHITECTURE:
Input layer:  x ∈ ℝⁿ (n features)
Hidden layers: learned representations
Output layer: ŷ (prediction)

FORWARD PASS:
For each layer l:
  Z[l] = W[l] @ A[l-1] + b[l]  ← linear
  A[l] = g(Z[l])                 ← activation

Where:
  W[l] ∈ ℝ^(n[l] × n[l-1])   weight matrix
  b[l] ∈ ℝ^n[l]               bias vector
  g    = activation function

ACTIVATION FUNCTIONS:

Sigmoid: σ(z) = 1/(1+e⁻ᶻ)
  → Output [0,1]
  → Problem: vanishing gradients! (σ'≈0 at ±∞)
  → Use only in output for binary classification

ReLU: g(z) = max(0, z)
  → g'(z) = 1 if z>0, else 0
  → No vanishing gradient for z>0!
  → Default choice for hidden layers
  → Problem: dying ReLU (z always <0)

Leaky ReLU: g(z) = max(0.01z, z)
  → Fixes dying ReLU (small gradient for z<0)

Tanh: g(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)
  → Output [-1,1] (zero-centered!)
  → Better than sigmoid, slower than ReLU

Softmax: g(zₖ) = e^zₖ / Σ e^zⱼ
  → Multi-class output
  → All outputs sum to 1.0

LOSS FUNCTIONS:
Binary:     L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
Multiclass: L = -Σ yₖ*log(ŷₖ)  (cross-entropy)
Regression: L = (1/2)(ŷ - y)²   (MSE)

BACKPROPAGATION:
Chain rule applied recursively!

∂L/∂W[l] = (1/m) * δ[l] @ A[l-1].T
∂L/∂b[l] = (1/m) * sum(δ[l])

Where δ[l] = backpropagated error:
δ[L] = ∂L/∂Z[L]          (output layer)
δ[l] = (W[l+1].T @ δ[l+1]) * g'(Z[l])

UPDATE:
W[l] := W[l] - α * ∂L/∂W[l]
b[l] := b[l] - α * ∂L/∂b[l]
"""

# 1. Activation functions + derivatives
class Activations:
    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(z):
        s = Activations.sigmoid(z)
        return s * (1 - s)  # beautiful!

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(z):
        return 1 - np.tanh(z)**2

    @staticmethod
    def softmax(z):
        # Numerically stable softmax
        exp_z = np.exp(z - z.max(axis=0,
                                   keepdims=True))
        return exp_z / exp_z.sum(axis=0,
                                  keepdims=True)

print("Activation function properties:")
z_test = np.array([-2, -1, 0, 1, 2])
print(f"ReLU(-2,0,2):    "
      f"{Activations.relu(z_test)}")
print(f"Sigmoid(-2,0,2): "
      f"{Activations.sigmoid(z_test).round(3)}")
print(f"Tanh(-2,0,2):    "
      f"{Activations.tanh(z_test).round(3)}")

# 2. Neural Network from scratch
class NeuralNetworkScratch:
    def __init__(self, layer_dims,
                  learning_rate=0.01,
                  epochs=1000,
                  activation='relu'):
        """
        layer_dims: [input, hidden1, ..., output]
        e.g. [784, 128, 64, 10]
        """
        self.layer_dims = layer_dims
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.params = {}
        self.costs = []
        self._init_params()

    def _init_params(self):
        """He initialization for ReLU"""
        np.random.seed(42)
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            n_curr = self.layer_dims[l]
            n_prev = self.layer_dims[l-1]
            # He init: scale by sqrt(2/n_prev)
            # Prevents vanishing/exploding gradients!
            self.params[f'W{l}'] = \
                np.random.randn(n_curr, n_prev) * \
                np.sqrt(2.0 / n_prev)
            self.params[f'b{l}'] = \
                np.zeros((n_curr, 1))

    def _activate(self, Z):
        if self.activation == 'relu':
            return Activations.relu(Z)
        elif self.activation == 'tanh':
            return Activations.tanh(Z)
        return Activations.sigmoid(Z)

    def _activate_deriv(self, Z):
        if self.activation == 'relu':
            return Activations.relu_deriv(Z)
        elif self.activation == 'tanh':
            return Activations.tanh_deriv(Z)
        return Activations.sigmoid_deriv(Z)

    def forward(self, X):
        """Forward pass — store cache for backprop"""
        cache = {'A0': X}
        L = len(self.layer_dims) - 1

        for l in range(1, L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            A_prev = cache[f'A{l-1}']

            Z = W @ A_prev + b
            cache[f'Z{l}'] = Z

            if l == L:  # output layer
                # Binary: sigmoid
                # Multi: softmax
                A = Activations.sigmoid(Z)
            else:
                A = self._activate(Z)

            cache[f'A{l}'] = A

        return cache

    def compute_loss(self, AL, Y):
        """Binary cross-entropy loss"""
        m = Y.shape[1]
        AL = np.clip(AL, 1e-8, 1-1e-8)
        loss = -(1/m) * np.sum(
            Y * np.log(AL) +
            (1-Y) * np.log(1-AL)
        )
        return loss

    def backward(self, cache, Y):
        """Backpropagation — chain rule!"""
        grads = {}
        L = len(self.layer_dims) - 1
        m = Y.shape[1]

        # Output layer gradient
        AL = cache[f'A{L}']
        # dL/dZ for sigmoid + binary cross-entropy
        # = ŷ - y (beautiful closed form!)
        dZ = AL - Y

        for l in reversed(range(1, L + 1)):
            A_prev = cache[f'A{l-1}']
            m_size = A_prev.shape[1]

            grads[f'dW{l}'] = (1/m_size) * \
                dZ @ A_prev.T
            grads[f'db{l}'] = (1/m_size) * \
                np.sum(dZ, axis=1,
                        keepdims=True)

            if l > 1:
                W = self.params[f'W{l}']
                Z_prev = cache[f'Z{l-1}']
                # Backprop through activation
                dA_prev = W.T @ dZ
                dZ = dA_prev * \
                    self._activate_deriv(Z_prev)

        return grads

    def update_params(self, grads):
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            self.params[f'W{l}'] -= \
                self.lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= \
                self.lr * grads[f'db{l}']

    def fit(self, X, Y):
        """X: (features, samples) column format"""
        for epoch in range(self.epochs):
            cache = self.forward(X)
            L = len(self.layer_dims) - 1
            loss = self.compute_loss(
                cache[f'A{L}'], Y
            )
            self.costs.append(loss)
            grads = self.backward(cache, Y)
            self.update_params(grads)

            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d}: "
                      f"loss={loss:.4f}")
        return self

    def predict(self, X, threshold=0.5):
        cache = self.forward(X)
        L = len(self.layer_dims) - 1
        AL = cache[f'A{L}']
        return (AL >= threshold).astype(int)

# 3. Train on classification data
print("\n=== TRAINING NEURAL NETWORK ===")
X_data, y_data = make_classification(
    n_samples=2000, n_features=20,
    n_informative=12, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_data,
    test_size=0.2, random_state=42
)

# Scale + reshape for column format
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).T
X_te_s = scaler.transform(X_te).T
y_tr_r = y_tr.reshape(1, -1)
y_te_r = y_te.reshape(1, -1)

# 3-layer network: 20 → 64 → 32 → 1
nn = NeuralNetworkScratch(
    layer_dims=[20, 64, 32, 1],
    learning_rate=0.01,
    epochs=1000,
    activation='relu'
)
nn.fit(X_tr_s, y_tr_r)

# Evaluate
train_pred = nn.predict(X_tr_s)
test_pred = nn.predict(X_te_s)
train_acc = accuracy_score(
    y_tr, train_pred.flatten()
)
test_acc = accuracy_score(
    y_te, test_pred.flatten()
)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# 4. Initialization importance
print("\n=== INITIALIZATION STRATEGIES ===")
print("""
Zero init: WRONG! All neurons learn same thing
           (symmetry breaking problem!)

Random (small): works but slow convergence

Xavier/Glorot (for tanh):
W ~ N(0, sqrt(1/n_prev))
→ Keeps variance stable through layers

He initialization (for ReLU):
W ~ N(0, sqrt(2/n_prev))  ← we used this!
→ Accounts for ReLU zeroing half the neurons
→ Best for deep ReLU networks

LeCun (for SELU):
W ~ N(0, sqrt(1/n_prev))
""")

# 5. Vanishing gradient problem
print("\n=== VANISHING GRADIENT PROBLEM ===")
print("""
Problem with sigmoid in deep networks:
σ'(z) = σ(z)(1-σ(z)) ≤ 0.25

With 10 layers:
∂L/∂W1 = product of 10 terms each ≤ 0.25
         ≈ 0.25^10 ≈ 0.000001

Gradient essentially ZERO by layer 1!
→ Early layers don't learn anything!

Solutions:
1. ReLU: gradient = 1 for z>0 (no shrinking!)
2. Batch Normalization (tomorrow!)
3. Residual connections (Skip connections)
4. Better initialization (He, Xavier)
5. Gradient clipping (for RNNs)
""")

# 6. Log to MLflow
mlflow.set_experiment("phase3_neural_networks")
with mlflow.start_run(
        run_name="NN_from_scratch"):
    mlflow.log_param("architecture",
                     str([20, 64, 32, 1]))
    mlflow.log_param("activation", "relu")
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 1000)
    mlflow.log_param("init", "He")
    mlflow.log_metric("train_acc", train_acc)
    mlflow.log_metric("test_acc", test_acc)
    mlflow.log_metric(
        "final_loss", nn.costs[-1]
    )
    print("\nNeural network logged to MLflow!")

print("\n" + "="*60)
print("Neural Networks — MASTERED! 🧠")
print("Week 2 Day 1 — COMPLETE!")
print("="*60)
