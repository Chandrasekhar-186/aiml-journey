## Top 7 Spark Optimization Rules

1. Single scan over multiple scans
   → Combine filters + aggs in one pass

2. F.when() over Python UDFs
   → Stays in Tungsten execution engine
   → 10-100× faster than UDF

3. Single groupBy with multiple aggs
   → One shuffle instead of many
   → Always list all aggs together

4. Broadcast join for small tables
   → No shuffle = massive speedup
   → Default threshold: 10MB
   → Tune: autoBroadcastJoinThreshold

5. Tune shuffle partitions
   → Default 200 often wrong
   → Target: ~128MB per partition
   → AQE handles this automatically!

6. Cache strategically
   → Only cache if reused 2+ times
   → Use MEMORY_AND_DISK in production
   → Always unpersist() when done!

7. Read explain plans
   → BroadcastHashJoin ✅ efficient
   → SortMergeJoin ⚠️ check if needed
   → Exchange ⚠️ shuffle happening
   → CartesianProduct 🚨 missing join key!

## Streaming Resilience
Checkpoint = exactly-once guarantee
Location: cloud storage (S3/ADLS/GCS)
Stores: offsets + state + metadata
Recovery: replay from last checkpoint

## Dijkstra Variants
Classic:      minimize total cost
Minimax:      minimize maximum cost
Max-prob:     maximize product of probs
All use:      heap + relaxation + visited
Key change:   update condition + heap value
