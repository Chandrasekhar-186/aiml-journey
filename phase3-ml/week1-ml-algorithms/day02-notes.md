## Logistic Regression — Key Math

Sigmoid: σ(z) = 1/(1+e^(-z)) → [0,1]
Cost:    J = -(1/m)Σ[y*log(h)+(1-y)*log(1-h)]
Gradient: SAME as linear regression!
          (1/m)Xᵀ(h-y)

Why log loss not MSE?
MSE + sigmoid = non-convex → local minima!
Log loss + sigmoid = convex → global min!

## Classification Metrics Cheat Sheet
Accuracy  = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)  ← minimize FP
Recall    = TP/(TP+FN)  ← minimize FN
F1        = 2*P*R/(P+R) ← balance P and R
ROC-AUC   = ranking quality (0.5-1.0)

When to use:
Balanced data:     Accuracy OK
Imbalanced data:   F1 or PR-AUC
Cost of FP high:   Precision (spam)
Cost of FN high:   Recall (cancer)
Ranking quality:   ROC-AUC

## Phase 2 → Phase 3 Connection
Linear Regression:  continuous output
Logistic Regression: sigmoid → probability
Neural Network:      layers of logistic units!
Deep Learning:       many layers, complex sigmoid

Everything in DL builds on today's math!

## Max Product — Dual Tracking
Track both cur_max AND cur_min
Because: negative × negative = positive!
At each step: candidates = (n, n×max, n×min)
New max = max of all 3 candidates
New min = min of all 3 candidates

## Phase 3 Week 1 Progress
Day 1: Linear Regression ✅
Day 2: Logistic Regression ✅
Day 3: Decision Trees (tomorrow)
Day 4: SVM + kernel trick
Day 5: Clustering algorithms
Day 6: PCA + dim reduction
Day 7: Week 1 review + project
