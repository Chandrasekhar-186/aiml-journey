# Phase 2 Day 3 — Spark Memory Management
# Date: April 13, 2026
# Goal: Never hit OOM errors in production!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

spark = SparkSession.builder \
    .appName("MemoryManagement") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.memoryFraction",
            "0.6") \
    .config("spark.memory.offHeap.enabled",
            "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

print("="*60)
print("Spark Memory Management — Complete Guide")
print("="*60)

"""
SPARK EXECUTOR MEMORY LAYOUT:

Total Executor Memory (e.g. 4GB)
├── Reserved Memory (300MB fixed)
│   └── Spark internal objects
│
├── User Memory (40% of remaining)
│   └── Your Python/Scala objects
│   └── UDF data structures
│   └── RDD partition metadata
│
└── Unified Memory (60% of remaining)
    ├── Execution Memory (dynamic)
    │   └── Shuffle buffers
    │   └── Sort buffers
    │   └── Aggregation hash maps
    │
    └── Storage Memory (dynamic)
        └── Cached RDDs/DataFrames
        └── Broadcast variables

KEY: Execution and Storage SHARE unified memory!
→ If execution needs more → evicts cached data
→ If storage needs more → can't evict execution
→ Set spark.memory.fraction (default 0.6)
→ Set spark.memory.storageFraction (default 0.5)
"""

# 1. Storage levels — know all for cert!
print("\n=== STORAGE LEVELS ===")
storage_levels = {
    "MEMORY_ONLY": "Store as deserialized Java objects in JVM. Fast but uses most memory.",
    "MEMORY_ONLY_SER": "Store as serialized (compact). Slower CPU but less memory.",
    "MEMORY_AND_DISK": "If doesn't fit in memory, spill to disk. Most common!",
    "MEMORY_AND_DISK_SER": "Serialized + disk spill. Best for large datasets.",
    "DISK_ONLY": "Store only on disk. Slow but saves memory.",
    "OFF_HEAP": "Store in off-heap memory (Tungsten). No GC pressure!"
}
for level, desc in storage_levels.items():
    print(f"\n  {level}:")
    print(f"    {desc}")

# 2. When to cache vs persist
data = [(i, f"model_{i%10}",
         float(i%100)) for i in range(50000)]
df = spark.createDataFrame(
    data, ["id", "model", "score"]
)

# cache() = MEMORY_ONLY
df.cache()
df.count()  # triggers caching
print(f"\nCached DataFrame!")
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Use cached DF multiple times (benefit!)
result1 = df.filter(F.col("score") > 80).count()
result2 = df.groupBy("model").count().show()
# Both reuse cached data!

# persist() with specific level
df2 = spark.createDataFrame(
    data, ["id", "model", "score"]
)
df2.persist(StorageLevel.MEMORY_AND_DISK)
df2.count()

# Always unpersist when done!
df.unpersist()
df2.unpersist()
print("DataFrames unpersisted — memory freed!")

# 3. Broadcast variables
print("\n=== BROADCAST VARIABLES ===")
# Large lookup table — broadcast to all executors
lookup = {i: f"category_{i%5}"
           for i in range(1000)}
broadcast_lookup = spark.sparkContext.broadcast(
    lookup
)

# Use in UDF (one of valid UDF use cases!)
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def get_category(id_val):
    return broadcast_lookup.value.get(
        id_val, "unknown"
    )

result = df.withColumn(
    "category", get_category(F.col("id"))
)
result.show(3)

# Clean up
broadcast_lookup.unpersist()
print("Broadcast variable unpersisted!")

# 4. Accumulators — distributed counters
print("\n=== ACCUMULATORS ===")
error_count = spark.sparkContext.accumulator(0)
null_count = spark.sparkContext.accumulator(0)

def count_issues(row):
    if row.score is None:
        null_count.add(1)
    if row.score < 0:
        error_count.add(1)
    return row

# Note: accumulators only reliable in actions!
df.foreach(count_issues)
print(f"Null scores:    {null_count.value}")
print(f"Negative scores: {error_count.value}")

# 5. OOM Prevention checklist
print("\n=== OOM PREVENTION CHECKLIST ===")
print("""
1. Always filter EARLY — reduce data volume
2. Select only needed columns (pruning)
3. Use persist(MEMORY_AND_DISK) not cache()
   for large DataFrames
4. Unpersist when DataFrame no longer needed
5. Avoid collect() on large DataFrames
   → Use take(n) or show() instead
6. Increase executor memory if needed
   spark.executor.memory = 8g
7. Enable off-heap memory for Tungsten
   spark.memory.offHeap.enabled = true
8. Check for data skew — one partition huge
9. Tune spark.sql.shuffle.partitions
   Default 200 — tune to 2-3x core count
10. Monitor via Spark UI → Storage tab!
""")
