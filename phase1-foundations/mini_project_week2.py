# Week 2 Mini Project — ML Experiment Tracker
# Date: March 26, 2026
# Stack: Pandas + Scikit-learn + MLflow + PyTorch
# This is your first COMPLETE project!

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report)

print("="*50)
print("Week 2 Mini Project: Wine Quality Classifier")
print("Comparing: RF vs GBM vs Neural Network")
print("="*50)

# 1. Load & explore dataset
data = load_wine()
X = pd.DataFrame(data.data,
                  columns=data.feature_names)
y = data.target

print(f"\nDataset: {X.shape[0]} wines, "
      f"{X.shape[1]} features")
print(f"Classes: {list(data.target_names)}")
print(f"\nFeature stats:\n{X.describe().round(2)}")

# 2. Feature engineering
X['alcohol_malic'] = X['alcohol'] / X['malic_acid']
X['color_hue_ratio'] = (X['color_intensity'] /
                          X['hue'])
print(f"\nFeatures after engineering: {X.shape[1]}")

# 3. Split + Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Train & compare all models — log to MLflow!
mlflow.set_experiment("wine_quality_comparison")

results = {}
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("features_engineered", 2)

        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds,
                       average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(model, name)

        results[name] = {"accuracy": acc, "f1": f1}
        print(f"\n{name}: Acc={acc:.4f} F1={f1:.4f}")

# 5. PyTorch Neural Network
class WineNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cpu')
nn_model = WineNN().to(device)
optimizer = torch.optim.Adam(
    nn_model.parameters(), lr=0.001
)
criterion = nn.CrossEntropyLoss()

X_tr = torch.FloatTensor(X_train_s)
y_tr = torch.LongTensor(y_train)
X_te = torch.FloatTensor(X_test_s)
y_te = torch.LongTensor(y_test)

with mlflow.start_run(run_name="NeuralNetwork"):
    mlflow.log_param("model", "PyTorch_NN")
    mlflow.log_param("architecture", "64-32-3")
    mlflow.log_param("epochs", 200)

    for epoch in range(200):
        nn_model.train()
        optimizer.zero_grad()
        out = nn_model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            mlflow.log_metric("loss",
                               loss.item(),
                               step=epoch)

    nn_model.eval()
    with torch.no_grad():
        preds_t = nn_model(X_te).argmax(1)
        acc = (preds_t == y_te).float().mean().item()
        f1 = f1_score(y_te.numpy(),
                       preds_t.numpy(),
                       average='weighted')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.pytorch.log_model(nn_model,
                               "neural_network")
    results["NeuralNetwork"] = {
        "accuracy": acc, "f1": f1
    }
    print(f"\nNeuralNetwork: "
          f"Acc={acc:.4f} F1={f1:.4f}")

# 6. Final comparison
print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
best = max(results, key=lambda x:
           results[x]['accuracy'])
for name, metrics in results.items():
    marker = " ← BEST" if name == best else ""
    print(f"{name:20s}: "
          f"Acc={metrics['accuracy']:.4f} "
          f"F1={metrics['f1']:.4f}{marker}")
print("\nAll runs logged to MLflow!")
print("Run 'mlflow ui' to compare visually!")
