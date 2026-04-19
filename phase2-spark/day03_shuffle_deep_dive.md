# Spark Shuffle — Complete Guide
Date: April 13, 2026

## What is a Shuffle?
Redistribution of data across partitions.
Required when data from different partitions
must be combined for the same key.

## What Triggers a Shuffle
→ groupBy() + aggregation
→ join() (unless broadcast)
→ distinct()
→ repartition()
→ orderBy() / sort()
→ cogroup(), groupWith()
→ intersection(), subtract()

## What DOESN'T Trigger a Shuffle
→ filter() — stays in same partition
→ select() / withColumn() — same partition
→ map() / flatMap() — same partition
→ union() — just concatenates partitions
→ coalesce() — reduces without shuffle
→ broadcast join — no shuffle!

## Shuffle Lifecycle
Stage 1 (Map side):
1. Each task processes its partition
2. Writes shuffle output to local disk
3. Sorts data by target partition
   (hash(key) % num_partitions)
4. Creates index files for fast lookup

Stage 2 (Reduce side):
1. Each task fetches data from ALL map tasks
2. Merges and sorts the fetched data
3. Performs final aggregation

## Why Shuffle is Expensive
→ Disk I/O (map output written to disk)
→ Network I/O (data transferred across nodes)
→ CPU (sort + merge operations)
→ Creates stage boundary (breaks pipelining)

## Shuffle Optimization Techniques

### 1. Reduce shuffle data volume
df.select("key", "value")  # drop cols before shuffle
  .groupBy("key")
  .agg(F.sum("value"))

### 2. Map-side aggregation (combiner)
# Spark does this automatically with DataFrame API!
# Partial aggregation happens before shuffle

### 3. Tune shuffle partitions
# Too many: small files, overhead
# Too few: large partitions, OOM risk
spark.conf.set(
    "spark.sql.shuffle.partitions", "200"
)
# Rule: 128MB-200MB per partition target size

### 4. AQE handles this automatically!
# With AQE enabled (Spark 3.2+):
# → Coalesces small partitions post-shuffle
# → No manual tuning needed!

### 5. Avoid shuffle entirely
# Use broadcast join for small tables
# Use coalesce() instead of repartition()
# Pre-partition data on join/group key

## Key Metric: Shuffle Read/Write
Check Spark UI → Stages tab
→ Shuffle Read: data fetched from other tasks
→ Shuffle Write: data written for other tasks
If Shuffle Write >> Shuffle Read → data skew!
