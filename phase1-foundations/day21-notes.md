## Spark Cert — Key Distinctions

repartition(n):  creates n partitions (SHUFFLE!)
coalesce(n):     reduces partitions (no shuffle)
→ Use coalesce to reduce, repartition to increase

cache():         MEMORY_ONLY storage
persist():       choose storage level
  MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY
→ Use cache() for frequently reused DataFrames

## Spark Job → Stage → Task hierarchy
Job:   triggered by one action (show, count)
Stage: set of transformations with no shuffle
Task:  one partition processed by one executor

## Union-Find — Two Optimizations
Path compression:  find() flattens the tree
Union by rank:     attach smaller tree to larger
→ Together achieve O(α(n)) ≈ O(1) amortized!

## Matrix Rotation — Visual Trick
90° clockwise = Transpose + Reverse each row
90° counter = Reverse each row + Transpose
180° = Reverse each row + Reverse columns

## Capstone Project Architecture
Data Generation → MLflow logging
      ↓
PySpark ingestion → Delta Lake storage
      ↓
Meta-model training → MLflow registry
      ↓
RAG system → answer questions about experiments
      ↓
CV component → classify performance charts
      ↓
Full integrated system → GitHub portfolio
