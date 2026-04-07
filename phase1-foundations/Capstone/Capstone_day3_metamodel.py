# Phase 1 Capstone — Day 3
# Meta-Model Training + MLflow Registry
# Date: April 4, 2026

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report
)
import xgboost as xgb

print("="*55)
print("Capstone Day 3: Meta-Model Training")
print("Predicting experiment success!")
print("="*55)

# 1. Load from Delta Lake
# (Using CSV since we're local — Delta in Databricks)
df = pd.read_csv('experiments_dataset.csv')
print(f"\nLoaded {len(df)} experiments")
print(f"Pass rate: {df['passed'].mean():.1%}")

# 2. Feature engineering for meta-model
le_model = LabelEncoder()
le_dataset = LabelEncoder()

df['model_type_enc'] = le_model.fit_transform(
    df['model_type']
)
df['dataset_enc'] = le_dataset.fit_transform(
    df['dataset']
)
df['n_estimators_filled'] = df[
    'n_estimators'].fillna(0)
df['lr_filled'] = df['learning_rate'].fillna(0)

feature_cols = [
    'model_type_enc', 'dataset_enc',
    'n_estimators_filled', 'lr_filled',
    'train_time'
]
X = df[feature_cols].values
y = df['passed'].astype(int).values

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# 4. Train multiple meta-models
mlflow.set_experiment("capstone_meta_models")

models = {
    "RF_meta": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGB_meta": xgb.XGBClassifier(
        n_estimators=100, random_state=42,
        verbosity=0
    ),
    "GBM_meta": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}

best_f1 = 0
best_model_name = ""
best_run_id = ""
results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model", name)
        mlflow.log_param("features",
                          feature_cols)

        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(
            X_test_s
        )[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, name)

        results[name] = {
            "acc": acc, "f1": f1
        }
        print(f"\n{name}: "
              f"Acc={acc:.4f} F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_run_id = run.info.run_id

# 5. PyTorch meta-model
class MetaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

X_tr_t = torch.FloatTensor(X_train_s)
y_tr_t = torch.FloatTensor(y_train)
X_te_t = torch.FloatTensor(X_test_s)
y_te_t = torch.FloatTensor(y_test)

nn_model = MetaNN()
optimizer = torch.optim.Adam(
    nn_model.parameters(), lr=0.001
)
criterion = nn.BCELoss()

with mlflow.start_run(run_name="PyTorch_meta") as run:
    mlflow.log_param("model", "PyTorch_MetaNN")
    mlflow.log_param("architecture", "5-32-16-1")

    for epoch in range(150):
        nn_model.train()
        optimizer.zero_grad()
        out = nn_model(X_tr_t).squeeze()
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            mlflow.log_metric(
                "loss", loss.item(),
                step=epoch
            )

    nn_model.eval()
    with torch.no_grad():
        preds_t = (
            nn_model(X_te_t).squeeze() > 0.5
        ).float()
        acc = (preds_t == y_te_t).float().mean()
        f1 = f1_score(
            y_te_t.numpy(), preds_t.numpy()
        )

    mlflow.log_metric("accuracy", acc.item())
    mlflow.log_metric("f1_score", f1)
    mlflow.pytorch.log_model(nn_model, "meta_nn")
    results["PyTorch_meta"] = {
        "acc": acc.item(), "f1": f1
    }
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = "PyTorch_meta"
        best_run_id = run.info.run_id

    print(f"\nPyTorch_meta: "
          f"Acc={acc:.4f} F1={f1:.4f}")

# 6. Register best model!
print(f"\n{'='*55}")
print(f"BEST MODEL: {best_model_name}")
print(f"F1 Score:   {best_f1:.4f}")
print(f"{'='*55}")

# Register to MLflow Model Registry
mlflow.register_model(
    f"runs:/{best_run_id}/{best_model_name}",
    "ExperimentSuccessPredictor"
)
print("\nBest model registered to MLflow Registry!")
print("Stage: None → Staging tomorrow!")

# 7. Final comparison table
print("\nFINAL LEADERBOARD:")
print(f"{'Model':20s} {'Accuracy':10s} {'F1':8s}")
print("-" * 40)
for name, m in sorted(
    results.items(),
    key=lambda x: x[1]['f1'],
    reverse=True
):
    marker = " ← BEST" if name == best_model_name \
             else ""
    print(f"{name:20s} {m['acc']:.4f}     "
          f"{m['f1']:.4f}{marker}")
