# Phase 2 Superday Stories — Updated
Date: May 8, 2026
These replace/enhance Phase 1 STAR stories!

## STORY 1: Technical Deep Dive
"Tell me about a complex technical problem
 you solved recently"

SITUATION:
Built a real-time ML model monitoring system
that needed to detect model drift in production.

TASK:
Design architecture that could process
streaming predictions, detect statistical
anomalies, and alert engineers within 1 minute.

ACTION:
→ Chose Lakehouse: Bronze (raw streaming) →
  Silver (quality + enrichment) →
  Gold (Z-score drift detection)
→ Used Z-score with rolling window of 100
  predictions — |z|>2.5 = anomaly
→ Integrated MLflow to tag DRIFT_DETECTED
  runs automatically
→ Added GraphFrames PageRank to identify
  most critical models for prioritization
→ Used Delta MERGE for idempotent writes
  to prevent duplicates on retry

RESULT:
→ Drift detected in <1 minute end-to-end
→ 7 Databricks competencies in one project
→ Production-grade: checkpoints, exactly-once,
  schema enforcement, time travel

## STORY 2: Performance Optimization
"Tell me about a time you optimized
 a slow system"

SITUATION:
Spark job taking 45 minutes on model metrics
aggregation.

TASK:
Get below 5 minutes without hardware changes.

ACTION:
→ Ran explain() — found SortMergeJoin on
  a 500MB lookup table
→ Added F.broadcast() hint → BroadcastHashJoin
  (no shuffle!)
→ Found groupBy called 3 separate times →
  Combined into single groupBy with all aggs
  (3 shuffles → 1 shuffle)
→ Enabled AQE for automatic partition coalescing
→ Added Z-ORDER on model_type for partition
  pruning (80% less data scanned!)

RESULT:
→ 45 minutes → 3.8 minutes (11x speedup!)
→ No hardware changes
→ Same result, dramatically less cost

## STORY 3: Learning + Adaptability
"Tell me about a time you had to learn
 something quickly"

SITUATION:
Needed to add drift detection to ML Monitor
project with no prior statistical monitoring
experience.

TASK:
Implement production-grade drift detection
using only Spark native capabilities.

ACTION:
→ Researched statistical drift methods:
  PSI, KS test, Z-score, CUSUM
→ Chose Z-score for Spark compatibility:
  computable with Window + stddev functions
  (no custom UDFs needed = Tungsten optimized!)
→ Implemented rolling 100-prediction window
→ Tested threshold |z|>2.5 (99.4% confidence)
→ Validated against synthetic drift scenarios

RESULT:
→ Working drift detection in 2 days
→ Zero Python UDFs — full Catalyst optimization
→ Generalizable to any numeric ML metric

## STORY 4: System Design
"Design a real-time ML feature store"

My answer framework:
1. Ingestion: Kafka → Spark Streaming
2. Storage: Delta Lake (online + offline)
   → Online: low-latency Delta reads
   → Offline: batch feature computation
3. Serving: MLflow Model Registry +
            Databricks Model Serving (REST)
4. Monitoring: THIS PROJECT (drift detection!)
5. Governance: Unity Catalog (access control)

Key design decisions:
→ Delta Lake: ACID + time travel + schema evol
→ Why not Redis for online? Delta with Photon
  is fast enough for most ML serving (<50ms)
→ Feature versioning: Delta time travel!
→ Backfill: same Spark pipeline, different dates

## BEHAVIORAL REFRESH
"Why Databricks specifically?"
Updated answer includes:
→ Spent 30 days going 10x deeper into their
  entire stack: Spark, Delta, MLflow, GraphFrames,
  Photon, Unity Catalog
→ Built production project using their exact
  architecture recommendations
→ The more I learn, the more I see how carefully
  designed each component is
→ I want to work on the tools I've been using
  every day for 58 days

"What's your greatest technical strength?"
→ I build things before I'm ready.
  I started using MLflow on Day 8.
  I built a streaming pipeline on Day 9.
  I connected every algorithm to Spark internals.
  I learn fastest by doing, not reading.
