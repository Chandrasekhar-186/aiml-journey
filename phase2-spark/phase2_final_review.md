# Phase 2 Final Review — Day 52
Date: May 2, 2026

## SPARK ARCHITECTURE — Can I explain from scratch?

Mental model:
Driver (brain) → DAGScheduler → TaskScheduler
                                    ↓
                         Cluster Manager
                                    ↓
                    Executors (muscle) × N

Execution flow:
1. Code creates logical plan (lazy!)
2. Action called → DAGScheduler builds DAG
3. DAGScheduler splits into stages at shuffles
4. TaskScheduler assigns tasks to executors
5. Executors run tasks in parallel
6. Results returned to driver

## CATALYST — 4 Phases
1. Analysis:       resolve names + types
2. Logical Opt:    pushdown, pruning, folding
3. Physical Plan:  join strategy selection
4. Code Gen:       Tungsten JVM bytecode

## MEMORY — 3 Regions
Reserved (300MB): Spark internals
User (40%):       Python objects + UDFs
Unified (60%):    Execution + Storage shared

## SHUFFLE — When triggered
groupBy, join (non-broadcast), distinct,
repartition, orderBy, cogroup, subtract,
intersection

NOT triggered:
filter, select, withColumn, union,
coalesce, map, flatMap, cache

## DELTA LAKE — Complete mental model
Files:        Parquet data files
Log:          _delta_log/ (JSON commits)
Checkpoint:   Every 10 commits (Parquet)
Time travel:  Read any past version
VACUUM:       Remove old files (7d default)
OPTIMIZE:     Compact small files
Z-ORDER:      Co-locate related data
MERGE:        Upsert (update + insert)
CDF:          Row-level change tracking
Liquid:       Incremental clustering

## STREAMING — Complete reference
Sources:    kafka, rate, socket, delta, file
Sinks:      delta, kafka, console, memory,
            foreach, foreachBatch
Modes:      append (new only, no agg!)
            complete (all rows, with agg)
            update (changed rows)
Triggers:   default, processingTime,
            once, availableNow, continuous
Checkpoint: ALWAYS required for production!
Watermark:  bounds state + handles late data

## OPTIMIZATION — Top 7 rules
1. Single scan (combine filters + aggs)
2. F.when() over Python UDFs
3. Single groupBy with all aggs
4. Broadcast small tables (< threshold)
5. Tune shuffle partitions (~128MB each)
6. Cache strategically (MEMORY_AND_DISK)
7. Read explain plans (find bottlenecks)

## MLFLOW — Complete stack
Tracking:  log_param, log_metric, log_artifact
Registry:  None→Staging→Production→Archived
Projects:  reproducible ML code packaging
Models:    unified model packaging format
Gateway:   unified LLM routing API
Served:    REST endpoint from registered model

## DATABRICKS SQL + PHOTON
DBSQL:  serverless, Photon engine, BI-optimized
Photon: C++ vectorized (not JVM!)
        5-10x faster for analytics
        SIMD: 8+ values per CPU instruction
Warehouses: Classic, Serverless, Pro

## CERTIFICATION CHECKLIST
□ Transformation vs Action (100% critical!)
□ cache() = lazy = MEMORY_ONLY
□ coalesce cannot increase partitions
□ append + aggregation = INVALID
□ Default shuffle partitions = 200
□ Delta checkpoint = every 10 commits
□ VACUUM default = 7 days
□ Broadcast threshold = 10MB
□ Semi join = left columns ONLY
□ AQE triggers at shuffle boundaries

## SELF TEST — Write from memory
1. Explain DAG vs RDD vs DataFrame
2. When does broadcast join NOT help?
3. Write a stateful streaming query
4. Explain Delta MERGE in one sentence
5. What is data skew + 2 fixes?
6. Difference between cache() + persist()?
7. When would you choose coalesce vs repartition?
8. What does CONVERT TO DELTA do?
9. Explain Z-score drift detection
10. What is the Lakehouse architecture?
