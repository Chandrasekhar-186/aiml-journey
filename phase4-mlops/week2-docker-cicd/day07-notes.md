## Databricks Model Serving
Managed REST endpoint for MLflow models
Auto-scales + scale-to-zero
Sizes: Small/Medium/Large/GPU
POST /serving-endpoints/{name}/invocations
Input: {"inputs": [[f1, f2, ...]]}
Output: {"predictions": [0, 1, ...]}

## Online vs Batch vs Streaming
Online:    <100ms, 1 record, user-triggered
           → Databricks Serving, FastAPI
Batch:     minutes, millions, scheduled
           → Spark Pandas UDF + Delta
Streaming: seconds, continuous, Kafka/Delta
           → Structured Streaming + UDF

## Feature Serving Patterns
Pre-computed: nightly batch → Redis/Delta
On-demand:    compute at request (fast ops!)
Streaming:    Kafka → Spark → Feature Store
Online Table: Delta → low-latency KV store
              single-digit ms lookups!

## Latency Percentiles (SLA)
p50: median (most requests)
p95: 95th percentile (worst 5%)
p99: 99th percentile (worst 1%)
p99.9: tail latency (SLA breaker!)
Target for serving: p99 < 200ms

## House Robber Pattern
Can't take adjacent elements
dp: prev2, prev1 = prev1, max(prev1, prev2+n)
Circular: run twice (skip first OR last)
Take max of both runs!

## Phase 4 Week 2 Progress
Day 6: Docker + FastAPI        ✅
Day 7: Model Serving           ✅ today
Day 8: Week 2 project          ← tomorrow
→ Comprehensive serving project!
