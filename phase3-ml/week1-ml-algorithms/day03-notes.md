## Decision Tree — Key Math

Gini Impurity:
G = 1 - Σpₖ²
G = 0:   pure (best!)
G = 0.5: maximum impurity

Information Gain:
IG = G(parent) - weighted_avg(G(children))
Choose split maximizing IG!

## Stopping Conditions
max_depth reached
min_samples_split not met
Node is pure (all same class)
No more features to split

## Random Forest — 3 Key Insights
1. Bootstrap sampling: ~63% unique per tree
   → 37% OOB = free validation!
2. Feature subsampling: sqrt(n) per split
   → Decorrelates trees → lower variance
3. Ensemble: majority vote → lower variance

Bias-Variance:
Single deep tree:  low bias, HIGH variance
RF average:        low bias, LOW variance ✅

## Feature Importance
= avg reduction in impurity per split
High importance: strong predictor
Suspiciously high: CHECK FOR DATA LEAKAGE!

## Interview Questions (answer cold!)
Q: Why does RF outperform single DT?
A: Reduces variance through bagging +
   feature subsampling → decorrelated trees

Q: What is OOB score?
A: Free cross-validation using the 37%
   of data each tree didn't see (bootstrap)

Q: Gini vs Entropy?
A: Similar results; Gini faster (no log),
   Entropy slightly better for imbalance

Q: How to fix DT overfitting?
A: max_depth, min_samples_split,
   min_samples_leaf, pruning, or use RF!

## Tree DSA → ML Connection
BST validation: bounds propagation
Build tree: preorder = root first
            inorder = left-root-right
DT splitting: same recursive structure!
