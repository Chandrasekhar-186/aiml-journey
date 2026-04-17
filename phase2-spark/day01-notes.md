# Phase 2 Notes — Day 1
## Spark Architecture Mental Model

Driver = brain (1 per application)
  → SparkContext: entry point
  → DAGScheduler: logical → physical plan
  → TaskScheduler: assigns tasks to executors

Executors = muscle (many per cluster)
  → Each has CPU cores + memory
  → Tasks run in parallel on cores
  → Cached data lives in executor memory

## Golden Rules
Rule 1: NEVER collect() large DataFrames
        → Crashes driver with OOM!
Rule 2: Filter EARLY — reduce data ASAP
Rule 3: Shuffle is expensive — minimize it
Rule 4: AQE handles most tuning automatically
Rule 5: DataFrame API > RDD API always

## Prefix Sum Pattern
Build running total as you scan
seen[prefix - k] = subarrays ending here = k
Classic: subarray sum, range sum queries

## Phase 2 Goal
Understand Spark so deeply that when
an interviewer asks "what happens when
you call groupBy()?" — you can explain:
  → DAGScheduler splits into 2 stages
  → Shuffle boundary created
  → Map-side partial aggregation
  → Data redistributed by hash(key)
  → Reduce-side final aggregation
  → AQE may adjust partition count
This level of depth = Databricks offer!
