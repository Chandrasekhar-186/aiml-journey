## Ensemble Methods — Interview Cheatsheet

Bagging (Random Forest):
→ Train trees in PARALLEL on random subsets
→ Average predictions → low variance
→ Good: noisy data, need fast training

Boosting (GBM/XGBoost/AdaBoost):
→ Train trees SEQUENTIALLY
→ Each tree corrects previous errors
→ Good: high accuracy, clean data

Stacking:
→ Train multiple models
→ Use their predictions as features
→ Meta-model makes final prediction

## XGBoost vs GBM
XGBoost adds:
  L1 + L2 regularization (prevents overfitting)
  Handles missing values automatically
  GPU acceleration
  Better default hyperparameters
→ Almost always use XGBoost over vanilla GBM!

## Spark MLlib Pipeline
Estimator → fit() → Transformer
Transformer → transform() → DataFrame
Pipeline → chains multiple stages

Key: Pipeline ensures same preprocessing
     applied to train AND test — no leakage!

## Heap — Python Implementation
import heapq
heapq.heappush(heap, val)  # push
heapq.heappop(heap)        # pop min
heap[0]                    # peek min
heapq.nlargest(k, iterable)  # top k largest
heapq.nsmallest(k, iterable) # top k smallest

Python only has MIN-heap!
For max-heap: push -val, pop -val

## Top K Pattern
K largest → min-heap size k → O(n log k)
K smallest → max-heap size k → O(n log k)
K most frequent → Counter + heap → O(n log k)
