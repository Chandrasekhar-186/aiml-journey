## PCA — Core Math Chain
Center → Covariance → Eigendecomposition
→ Sort by eigenvalue → Project to top k

Covariance: Σ = (1/m) XᵀX
Eigenvectors: directions of max variance
Eigenvalues:  variance along each direction

## Choosing K
Plot cumulative explained variance
k = first point where cumsum ≥ 95%
sklearn shortcut: PCA(n_components=0.95)

## PCA Limitations
Linear only (no curves!)
Sensitive to scale → ALWAYS standardize first!
Components hard to interpret

## t-SNE Key Facts
Local structure preservation only
Non-deterministic (random_state for reproducibility)
Cannot project new points!
Perplexity: 5-50 (default 30)
Always pre-reduce with PCA for speed

## UMAP vs t-SNE
UMAP:  faster, global+local, new points ✅
t-SNE: slower, local only, no new points ❌

## Dimensionality Reduction Interview Q
Q: When use PCA vs t-SNE?
A: PCA: before ML model (preprocessing)
        fast, linear, new points OK
   t-SNE: visualization ONLY
          beautiful clusters, no pipeline use

Q: Why standardize before PCA?
A: Features with larger scale dominate
   variance → PCA finds scale, not structure!
   Standardize → all features contribute equally

## Phase 3 Week 1 — Almost Complete!
Day 1: Linear Regression    ✅
Day 2: Logistic Regression  ✅
Day 3: Decision Trees + RF  ✅
Day 4: SVM + kernel trick   ✅
Day 5: Clustering           ✅
Day 6: PCA + t-SNE + UMAP   ✅ today
Day 7: Week 1 review + project start
