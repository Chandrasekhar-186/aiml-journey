# Phase 3 Day 3 — Decision Trees + Random Forest
# Date: May 11, 2026
# The most interpretable ML model!

import numpy as np
from sklearn.datasets import (
    make_classification, load_iris
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, classification_report
)
from collections import Counter
import mlflow

print("="*60)
print("Decision Trees + Random Forest")
print("="*60)

"""
DECISION TREE — COMPLETE MATH

A Decision Tree splits data recursively
by asking binary questions:
"Is feature X > threshold T?"

At each node, choose split that maximizes
INFORMATION GAIN (or minimizes impurity)

IMPURITY MEASURES:

1. Gini Impurity (CART algorithm):
   Gini = 1 - Σ pₖ²
   where pₖ = proportion of class k
   Gini = 0: pure node (all same class)
   Gini = 0.5: maximum impurity (50/50 split)

2. Entropy (ID3, C4.5 algorithms):
   H = -Σ pₖ * log₂(pₖ)
   Entropy = 0: pure node
   Entropy = 1: maximum impurity

3. Information Gain:
   IG = Entropy(parent) - weighted avg Entropy(children)
   Choose split that MAXIMIZES IG!

WHY Gini vs Entropy?
→ Both give similar results
→ Gini: slightly faster (no log computation)
→ Entropy: slightly better for unbalanced classes
→ sklearn default: Gini

STOPPING CONDITIONS:
→ max_depth reached
→ min_samples_split not met
→ Pure node (all same class)
→ No more features to split on
"""

# 1. Implement Decision Tree node
class Node:
    def __init__(self, feature=None,
                  threshold=None, left=None,
                  right=None, value=None):
        self.feature = feature      # split feature
        self.threshold = threshold  # split threshold
        self.left = left           # left subtree
        self.right = right         # right subtree
        self.value = value         # leaf prediction

class DecisionTreeScratch:
    def __init__(self, max_depth=10,
                  min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def gini(self, y):
        """Gini impurity"""
        if len(y) == 0: return 0
        counts = Counter(y)
        probs = [c/len(y)
                  for c in counts.values()]
        return 1 - sum(p**2 for p in probs)

    def best_split(self, X, y):
        """Find best feature + threshold"""
        best_gain = -1
        best_feat = best_thresh = None
        parent_gini = self.gini(y)
        m, n = X.shape

        for feat in range(n):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                # Split data
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if (sum(left_mask) < self.min_samples
                        or sum(right_mask) 
                        self.min_samples):
                    continue

                # Weighted child gini
                n_l = sum(left_mask)
                n_r = sum(right_mask)
                g_l = self.gini(y[left_mask])
                g_r = self.gini(y[right_mask])
                weighted = (n_l*g_l + n_r*g_r)/m

                # Information gain
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def build(self, X, y, depth=0):
        """Recursively build tree"""
        # Stopping conditions
        if (depth >= self.max_depth or
                len(y) < self.min_samples or
                len(np.unique(y)) == 1):
            # Leaf: predict majority class
            return Node(value=Counter(y)
                        .most_common(1)[0][0])

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return Node(value=Counter(y)
                        .most_common(1)[0][0])

        # Recurse on children
        mask = X[:, feat] <= thresh
        left = self.build(X[mask], y[mask],
                           depth+1)
        right = self.build(X[~mask], y[~mask],
                            depth+1)
        return Node(feat, thresh, left, right)

    def fit(self, X, y):
        self.root = self.build(X, y)
        return self

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([
            self.predict_one(x, self.root)
            for x in X
        ])

# 2. Test on Iris dataset
print("\n=== DECISION TREE ON IRIS ===")
iris = load_iris()
X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target,
                     test_size=0.2,
                     random_state=42)

# Our scratch implementation
dt_scratch = DecisionTreeScratch(max_depth=5)
dt_scratch.fit(X_train, y_train)
y_pred_scratch = dt_scratch.predict(X_test)
acc_scratch = accuracy_score(
    y_test, y_pred_scratch
)
print(f"Scratch DT accuracy: {acc_scratch:.4f}")

# sklearn comparison
dt_sk = DecisionTreeClassifier(
    max_depth=5, random_state=42
)
dt_sk.fit(X_train, y_train)
acc_sk = accuracy_score(
    y_test, dt_sk.predict(X_test)
)
print(f"sklearn DT accuracy: {acc_sk:.4f}")

# 3. Decision Tree hyperparameters
print("\n=== HYPERPARAMETER GUIDE ===")
print("""
max_depth:
→ Low: underfitting (too simple)
→ High: overfitting (memorizes training data)
→ Tune with cross-validation!

min_samples_split:
→ Minimum samples to split a node
→ Higher: more regularization

min_samples_leaf:
→ Minimum samples in leaf node
→ Higher: smoother decision boundary

criterion:
→ "gini" (default): faster
→ "entropy": slightly better for imbalance

max_features:
→ "sqrt": good for classification
→ "log2": another common choice
→ None: use all features (may overfit!)
""")

# 4. Random Forest — Bagging + Feature Sampling
print("\n=== RANDOM FOREST ===")
print("""
Why does a single Decision Tree overfit?
→ Grows until perfectly fitting training data
→ High variance — sensitive to data changes

Random Forest solution: ENSEMBLE!
→ Train N trees (e.g. 100-500)
→ Each tree on BOOTSTRAP SAMPLE (with replacement)
→ Each split considers RANDOM SUBSET of features
→ Predict: majority vote (classification)
           or average (regression)

Two sources of randomness:
1. Bootstrap sampling: each tree sees ~63%
   of unique training examples
   (37% = out-of-bag = free validation!)

2. Feature subsampling: each split considers
   sqrt(n_features) random features
   → Decorrelates trees!
   → Each tree learns different aspects

Why this works (Bias-Variance):
→ Deep trees: low bias, HIGH variance
→ Average of many: low bias, LOW variance
→ "Wisdom of crowds" effect!

OOB (Out-of-Bag) validation:
→ Each tree trained on ~63% of data
→ Remaining 37% = OOB samples
→ Evaluate tree on its OOB = free CV!
→ oob_score=True in sklearn
""")

# 5. Random Forest implementation
print("\n=== RANDOM FOREST TRAINING ===")
X, y = make_classification(
    n_samples=2000, n_features=20,
    n_informative=12, random_state=42
)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2,
                     random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,       # grow fully!
    max_features="sqrt",  # key for RF!
    min_samples_leaf=1,
    bootstrap=True,       # bagging!
    oob_score=True,       # free validation!
    n_jobs=-1,            # use all CPUs
    random_state=42
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"RF Test accuracy: {rf_acc:.4f}")
print(f"RF OOB accuracy:  {rf.oob_score_:.4f}")

# 6. Feature importance
print("\n=== FEATURE IMPORTANCE ===")
importances = rf.feature_importances_
top_features = np.argsort(importances)[::-1][:5]
print("Top 5 features by importance:")
for rank, feat in enumerate(top_features, 1):
    print(f"  {rank}. Feature {feat:2d}: "
          f"{importances[feat]:.4f}")

print("""
Feature importance = average reduction in
impurity (Gini/Entropy) across all trees
when that feature is used for splitting.

Use for:
→ Feature selection (drop low-importance)
→ Model interpretability
→ Understanding what drives predictions
→ Catching data leakage!
   (suspiciously high importance = leak?)
""")

# 7. Cross-validation
cv_scores = cross_val_score(
    rf, X, y, cv=5, scoring='accuracy'
)
print(f"\n5-fold CV: {cv_scores.mean():.4f}"
      f" ± {cv_scores.std():.4f}")

# 8. Log to MLflow
mlflow.set_experiment("phase3_trees_rf")
with mlflow.start_run(run_name="RF_baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_features", "sqrt")
    mlflow.log_param("bootstrap", True)
    mlflow.log_metric("test_accuracy", rf_acc)
    mlflow.log_metric("oob_score",
                       rf.oob_score_)
    mlflow.log_metric("cv_mean",
                       cv_scores.mean())
    mlflow.log_metric("cv_std",
                       cv_scores.std())
    print("\nRandom Forest logged to MLflow!")

print("\n" + "="*60)
print("Decision Trees + RF — MASTERED! 🌳")
print("="*60)
