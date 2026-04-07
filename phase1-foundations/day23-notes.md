## Meta-Learning Concept
Meta-model = model that predicts model performance
Input features: model type, dataset, hyperparams
Output: will this experiment succeed?
Use case: AutoML, hyperparameter search pruning

## MLflow Model Registry Workflow
1. Train models → log to MLflow
2. Compare runs in UI
3. Register best model:
   mlflow.register_model(run_uri, model_name)
4. Transition: None → Staging → Production
5. Load in production:
   mlflow.sklearn.load_model("models:/name/Production")

## DP on Strings Pattern
dp[i] = answer for first i characters
Base: dp[0] = 1 (empty string)
Transition: check last 1 or 2 characters
Examples: decode ways, word break, palindromes

## Greedy vs DP
Use Greedy when:
→ Local optimal choice leads to global optimal
→ No need to reconsider past decisions
→ Examples: jump game, interval scheduling

Use DP when:
→ Need to consider all subproblems
→ Overlapping subproblems exist
→ Examples: coin change, LIS, decode ways

## Bias-Variance Tradeoff
High bias (underfitting):
→ Model too simple, misses patterns
→ Fix: more features, complex model

High variance (overfitting):
→ Model memorizes training data
→ Fix: more data, regularization, dropout
