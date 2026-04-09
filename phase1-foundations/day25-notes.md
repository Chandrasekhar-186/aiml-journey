## Capstone Architecture — Final Summary

Layer    |     Component      |        Purpose
─────────────────────────────────────────────
Data Eng:  |   PySpark + Delta Lake |  Store+query
ML:        |   XGBoost + PyTorch NN |  Predict success
MLOps:     |  MLflow Registry       | Track+version
GenAI:     |   RAG (LangChain+FAISS) | NL queries
CV:        |  CNN + YOLOv8          | Chart classify
Infra:     |   Docker + Git          | Deploy+version

## HashSet Trick for Consecutive Sequence

Only start counting from sequence START
(n-1 not in set → n is start of sequence)
This avoids O(n²) by ensuring each number
is visited at most twice → O(n)!

## 0/1 Knapsack Pattern

"Can I reach exactly target?"
dp = set of achievable sums
For each item: dp |= {s + item for s in dp}
O(n * target) time, O(target) space

## Behavioral — 3 Questions to Always Ask

1. "What does success look like in 90 days?"
   → Shows you think about impact
2. "How does the team prioritize features?"
   → Shows product thinking
3. "What separates top performers here?"
   → Shows growth mindset
