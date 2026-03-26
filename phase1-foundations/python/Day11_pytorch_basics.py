# Day 11 — PyTorch Foundations
# Date: March 23, 2026
# Critical: Required by Databricks for AI/ML roles!

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow

# 1. Tensors — PyTorch's core data structure
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.zeros(2, 3)
z = torch.randn(3, 3)  # random normal

print(f"Tensor x:\n{x}")
print(f"Shape: {x.shape}")
print(f"Device: {x.device}")
print(f"Dtype: {x.dtype}")

# 2. Tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"Add:     {a + b}")
print(f"Dot:     {torch.dot(a, b)}")
print(f"Mean:    {a.mean()}")

# Move to GPU if available
device = torch.device('cuda' if
                       torch.cuda.is_available()
                       else 'cpu')
print(f"Using device: {device}")

# 3. Autograd — automatic differentiation!
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x  # y = x³ + 2x
y.backward()         # compute dy/dx
print(f"dy/dx at x=2: {x.grad}")  # 3x²+2 = 14

# 4. Build first neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size,
                 hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size,
                                 hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size,
                                 output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# 5. Train on breast cancer data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Initialize model
model = SimpleNN(30, 64, 1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop — log to MLflow!
mlflow.set_experiment("pytorch_neural_network")

with mlflow.start_run(run_name="SimpleNN_v1"):
    mlflow.log_param("hidden_size", 64)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("optimizer", "Adam")

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t).squeeze()
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test_t).squeeze()
                preds = (test_out > 0.5).float()
                acc = (preds == y_test_t).float().mean()
            mlflow.log_metric("loss", loss.item(),
                               step=epoch)
            mlflow.log_metric("accuracy", acc.item(),
                               step=epoch)
            print(f"Epoch {epoch+1}: "
                  f"Loss={loss.item():.4f} "
                  f"Acc={acc.item():.4f}")

    # Save PyTorch model to MLflow
    mlflow.pytorch.log_model(model, "neural_network")
    print("PyTorch model logged to MLflow!")
