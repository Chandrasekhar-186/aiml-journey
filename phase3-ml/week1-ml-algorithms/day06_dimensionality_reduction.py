# Phase 3 Day 6 — Dimensionality Reduction
# Date: May 14, 2026
# PCA + t-SNE + UMAP — reduce dimensions!

import numpy as np
from sklearn.datasets import (
    load_iris, load_digits, make_classification
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

print("="*60)
print("Dimensionality Reduction — Complete Guide")
print("="*60)

"""
DIMENSIONALITY REDUCTION — WHY?

Problems with high-dimensional data:
1. Curse of dimensionality:
   → Distance metrics break down in high-dim
   → All points become equally distant!
2. Computational cost: O(d²) or worse
3. Visualization: can't plot >3D
4. Overfitting: too many features

Two approaches:
Feature Selection:  keep subset of features
Feature Extraction: create new features
                    (PCA, t-SNE, Autoencoders)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PCA — PRINCIPAL COMPONENT ANALYSIS

Core idea: find directions of maximum variance

Steps:
1. Center data: X_c = X - mean(X)
2. Compute covariance matrix:
   Σ = (1/m) * X_cᵀ * X_c
3. Eigendecomposition:
   Σ = V * Λ * Vᵀ
   V = eigenvectors (principal components)
   Λ = eigenvalues (variance explained)
4. Sort by eigenvalue (descending)
5. Project: X_reduced = X_c * V_k
   (keep top k eigenvectors)

Geometric interpretation:
→ Each eigenvector = direction in feature space
→ Eigenvalue = variance along that direction
→ PCA rotates coordinates to align with
  directions of maximum variance!

Choosing k:
→ Explained variance ratio: cumsum(λ)/sum(λ)
→ Keep k where cumsum > 95% (or 99%)

Limitations:
→ Linear only (can't capture curves!)
→ Global structure, not local
→ Hard to interpret components
"""

# 1. PCA from scratch
class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Center data
        self.mean = X.mean(axis=0)
        X_c = X - self.mean

        # Covariance matrix
        cov = np.cov(X_c.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = \
            np.linalg.eig(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real

        # Store top k components
        self.components = \
            eigenvectors[:, :self.n_components].T

        # Explained variance ratio
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = \
            eigenvalues[:self.n_components] / \
            total_var

        return self

    def transform(self, X):
        X_c = X - self.mean
        return X_c @ self.components.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# 2. Test on Iris
print("\n=== PCA ON IRIS ===")
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Our scratch PCA
pca_scratch = PCAScratch(n_components=2)
X_2d = pca_scratch.fit_transform(X_scaled)

print(f"Original shape:  {X_scaled.shape}")
print(f"Reduced shape:   {X_2d.shape}")
print(f"Explained var:   "
      f"{pca_scratch.explained_variance_ratio_}")
print(f"Total explained: "
      f"{pca_scratch.explained_variance_ratio_.sum():.3f}")

# sklearn comparison
pca_sk = PCA(n_components=2)
X_sk = pca_sk.fit_transform(X_scaled)
print(f"\nsklearn explained: "
      f"{pca_sk.explained_variance_ratio_}")

# 3. Choosing number of components
print("\n=== CHOOSING K COMPONENTS ===")
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(
    pca_full.explained_variance_ratio_
)
print("Cumulative explained variance:")
for i, cv in enumerate(cumvar, 1):
    bar = "█" * int(cv * 30)
    print(f"  k={i}: {cv:.3f} {bar}")

# Find k for 95% variance
k_95 = np.argmax(cumvar >= 0.95) + 1
print(f"\nComponents for 95% variance: {k_95}")

# 4. PCA for ML pipeline
print("\n=== PCA IN ML PIPELINE ===")
X_full, y_full = make_classification(
    n_samples=1000, n_features=50,
    n_informative=15, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_full,
    test_size=0.2, random_state=42
)

# Without PCA
pipe_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
pipe_no_pca.fit(X_tr, y_tr)
acc_no_pca = accuracy_score(
    y_te, pipe_no_pca.predict(X_te)
)

# With PCA
pipe_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # 95% var!
    ('clf', LogisticRegression(max_iter=1000))
])
pipe_pca.fit(X_tr, y_tr)
acc_pca = accuracy_score(
    y_te, pipe_pca.predict(X_te)
)
n_comp = pipe_pca.named_steps['pca']\
    .n_components_

print(f"Without PCA: {acc_no_pca:.4f} "
      f"(50 features)")
print(f"With PCA:    {acc_pca:.4f} "
      f"({n_comp} features)")
print(f"Compression: 50 → {n_comp} features "
      f"({n_comp/50:.0%})")

# 5. t-SNE
print("\n=== t-SNE ===")
print("""
t-SNE = t-distributed Stochastic
        Neighbor Embedding

Purpose: VISUALIZATION only (not ML pipeline!)
Goal:    preserve LOCAL structure
         (nearby points stay nearby)

Algorithm:
1. Compute pairwise similarities in high-dim
   using Gaussian: p(j|i) ∝ exp(-||xᵢ-xⱼ||²/2σ²)
2. In low-dim (2D/3D), use t-distribution:
   q(j|i) ∝ (1 + ||yᵢ-yⱼ||²)⁻¹
3. Minimize KL divergence: KL(P||Q)
   using gradient descent

WHY t-distribution in low-dim?
→ Heavier tails than Gaussian
→ Moderate distances pushed apart
→ Creates cleaner cluster separation
→ Avoids "crowding problem"

Key parameters:
perplexity: balances local vs global structure
            (usually 5-50, default 30)
n_iter:     optimization iterations (default 1000)

IMPORTANT:
→ t-SNE is NON-DETERMINISTIC!
→ Different runs → different layouts
→ Cannot project NEW points!
→ Use PCA first for speed (pre-reduce to 50D)
→ Distances NOT meaningful between clusters!
""")

# t-SNE on digits
digits = load_digits()
X_digits = digits.data[:500]  # small subset
y_digits = digits.target[:500]

# Pre-reduce with PCA first (common practice!)
pca_50 = PCA(n_components=50)
X_pca = pca_50.fit_transform(
    StandardScaler().fit_transform(X_digits)
)

tsne = TSNE(n_components=2, perplexity=30,
             n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

print(f"Digits shape:    {X_digits.shape}")
print(f"After PCA(50):   {X_pca.shape}")
print(f"After t-SNE:     {X_tsne.shape}")
print("t-SNE visualization ready!")

# 6. UMAP overview
print("\n=== UMAP ===")
print("""
UMAP = Uniform Manifold Approximation
       and Projection

Advantages over t-SNE:
→ MUCH faster (especially large datasets)
→ Preserves GLOBAL structure better
→ CAN project new points (transform())
→ Can be used in ML pipelines!
→ Works for more than 2 dimensions

Math: based on Riemannian geometry
      and topological data analysis

When to use:
→ Large datasets (t-SNE too slow)
→ Need to project new points
→ Want global structure preserved

# pip install umap-learn
# import umap
# reducer = umap.UMAP(n_components=2)
# X_umap = reducer.fit_transform(X)
# X_new = reducer.transform(X_test)  # new points!
""")

# 7. Comparison summary
print("\n=== COMPARISON ===")
print("""
           PCA      t-SNE    UMAP
Linear:    YES      NO       NO
Preserves: Global   Local    Both
Speed:     FAST     SLOW     Medium
New pts:   YES      NO       YES
Stochastic:NO       YES      YES
Use for:   Pipeline  Viz     Viz/Pipeline
Dims out:  Any      2-3      Any
""")

# Log to MLflow
mlflow.set_experiment("phase3_dim_reduction")
with mlflow.start_run(
        run_name="PCA_comparison"):
    mlflow.log_param("dataset", "classification_50d")
    mlflow.log_param("pca_variance", 0.95)
    mlflow.log_metric(
        "acc_no_pca", acc_no_pca
    )
    mlflow.log_metric("acc_with_pca", acc_pca)
    mlflow.log_metric("n_components", n_comp)
    mlflow.log_metric(
        "compression_ratio", n_comp/50
    )
    print("\nDimensionality reduction logged!")

print("\n" + "="*60)
print("Dimensionality Reduction — MASTERED! 📐")
print("="*60)
