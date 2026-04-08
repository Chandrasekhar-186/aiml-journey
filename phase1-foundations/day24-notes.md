## Feature Store Architecture
Online store:  Redis/DynamoDB → <10ms latency
Offline store: Delta Lake → batch training
Serving API:   REST endpoint → real-time inference
Pipeline:      Spark Streaming → compute features
               → write to both stores simultaneously

## RAG on Experiment Data
Convert structured data → natural language docs
Embed with sentence-transformers
Store in FAISS vector index
Query: "Which XGBoost experiments had >90% acc?"
→ semantic search → retrieve relevant rows
→ better than SQL for fuzzy/natural queries!

## Max Product Subarray — Key Insight
Negative × Negative = Positive!
Must track BOTH min and max products
When hit 0: reset both to 1 (fresh start)
cur_max = max(n, cur_max*n, cur_min*n)
cur_min = min(n, cur_max*n, cur_min*n)

## Interval Problems — Greedy Rules
Sort by END time → maximize remaining space
Sort by START time → detect overlaps
Remove overlapping → keep earlier end time
Merge overlapping → extend current interval

## System Design — Feature Store
Key trade-off:
Online store: fast reads, expensive, limited size
Offline store: slow reads, cheap, unlimited size
Solution: dual-write to both simultaneously!
```
