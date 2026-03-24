## Model Evaluation — Complete Reference

Metric      Formula                When to use
─────────────────────────────────────────────
Accuracy    TP+TN / Total          Balanced classes
Precision   TP / (TP+FP)          Minimize false alarms
Recall      TP / (TP+FN)          Minimize missed cases
F1          2*P*R / (P+R)         Imbalanced classes
ROC AUC     Area under ROC curve  Ranking/threshold-free

## GridSearchCV vs RandomizedSearchCV
GridSearch:     tries ALL combinations → thorough, slow
RandomizedSearch: samples N combinations → fast, good enough
Use GridSearch for small param spaces (<20 combos)
Use RandomizedSearch for large spaces (deep learning)

## PySpark UDF Warning!
UDFs break Spark's Catalyst optimizer
→ Spark can't optimize Python UDFs
→ Use built-in F.functions whenever possible
→ Only use UDF when no built-in exists
→ If must use UDF → use Pandas UDF (vectorized)

## Graph DFS Template
def dfs(r, c):
    if out_of_bounds or visited: return
    mark_visited(r, c)
    for each neighbor:
        dfs(neighbor)

Works for: islands, flood fill, connected
components, path finding
