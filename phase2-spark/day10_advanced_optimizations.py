# Phase 2 Day 10 — Advanced Spark Optimizations
# Date: April 20, 2026
# Master-level optimization techniques!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import mlflow

spark = SparkSession.builder \
    .appName("AdvancedOptimizations") \
    .config("spark.sql.adaptive.enabled",
            "true") \
    .config("spark.sql.adaptive.coalescePartitions"
            ".enabled", "true") \
    .config("spark.sql.adaptive.skewJoin"
            ".enabled", "true") \
    .getOrCreate()

print("="*60)
print("Advanced Spark Optimizations")
print("="*60)

# 1. Query optimization patterns
data = [(i, f"user_{i%1000}",
         f"model_{i%10}",
         float(i%100), i%5)
        for i in range(500000)]
df = spark.createDataFrame(
    data, ["id", "user", "model",
           "score", "segment"]
)

# BAD: Multiple scans
print("\n=== AVOID MULTIPLE SCANS ===")
bad_count = df.filter(
    F.col("score") > 80).count()
bad_avg = df.filter(
    F.col("score") > 80).agg(
    F.avg("score")).collect()[0][0]
# ❌ Scans df TWICE!

# GOOD: Single scan with agg
good = df.filter(F.col("score") > 80).agg(
    F.count("*").alias("count"),
    F.avg("score").alias("avg_score")
).collect()[0]
# ✅ Single scan!
print(f"Count: {good.count}, "
      f"Avg: {good.avg_score:.2f}")

# 2. Avoid UDFs — use built-ins
print("\n=== AVOID UDFs ===")
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# BAD: Python UDF breaks Tungsten!
@udf(returnType=StringType())
def grade_udf(score):
    if score >= 90: return "A"
    elif score >= 80: return "B"
    return "C"
# df.withColumn("grade", grade_udf("score"))
# ❌ Slow — crosses JVM/Python boundary!

# GOOD: Use F.when() — stays in JVM!
df_graded = df.withColumn("grade",
    F.when(F.col("score") >= 90, "A")
     .when(F.col("score") >= 80, "B")
     .otherwise("C")
)
# ✅ Stays in Tungsten execution engine!
df_graded.show(3)

# 3. Efficient aggregations
print("\n=== EFFICIENT AGGREGATIONS ===")

# BAD: Multiple groupBy passes
seg_count = df.groupBy("segment").count()
seg_avg = df.groupBy("segment").avg("score")
# ❌ Two shuffles!

# GOOD: Single groupBy with multiple aggs
seg_stats = df.groupBy("segment").agg(
    F.count("*").alias("count"),
    F.avg("score").alias("avg"),
    F.max("score").alias("max"),
    F.min("score").alias("min"),
    F.stddev("score").alias("std"),
    F.percentile_approx(
        "score", 0.5
    ).alias("median")
)
# ✅ Single shuffle!
seg_stats.show()

# 4. Broadcast threshold tuning
print("\n=== BROADCAST THRESHOLD ===")
print(f"Current broadcast threshold: "
      f"{spark.conf.get('spark.sql.autoBroadcastJoinThreshold')}")

# Default 10MB — increase for larger lookups
spark.conf.set(
    "spark.sql.autoBroadcastJoinThreshold",
    str(50 * 1024 * 1024)  # 50MB
)
print("Broadcast threshold increased to 50MB!")

# Explicit broadcast hint
small_df = spark.createDataFrame(
    [(i, f"segment_{i}") for i in range(5)],
    ["segment_id", "segment_name"]
)
joined = df.join(
    F.broadcast(small_df),
    df.segment == small_df.segment_id
)
joined.explain()

# 5. Partition optimization
print("\n=== PARTITION OPTIMIZATION ===")

# Check current partitions
print(f"Default partitions: "
      f"{df.rdd.getNumPartitions()}")

# Tune shuffle partitions for dataset size
# Rule: ~128MB per partition
# 500k rows × ~100 bytes ≈ 50MB
# → 1-2 partitions optimal
spark.conf.set(
    "spark.sql.shuffle.partitions", "4"
)

result = df.groupBy("model").agg(
    F.avg("score")
).orderBy("model")
result.show()
print(f"Result partitions: "
      f"{result.rdd.getNumPartitions()}")

# 6. Caching strategy
print("\n=== CACHING STRATEGY ===")
from pyspark.storagelevel import StorageLevel

# Cache frequently reused DataFrame
df.persist(StorageLevel.MEMORY_AND_DISK)
df.count()  # trigger caching

# Multiple operations reuse cache
op1 = df.filter(F.col("score") > 90).count()
op2 = df.groupBy("model").count().collect()
op3 = df.agg(F.avg("score")).collect()
print(f"Op1: {op1} | Op2: {len(op2)} | "
      f"Op3: {op3[0][0]:.2f}")

df.unpersist()
print("Cache freed!")

# 7. Explain plan analysis
print("\n=== READING EXPLAIN PLANS ===")
complex_query = (df
    .filter(F.col("score") > 75)
    .join(F.broadcast(small_df),
          df.segment == small_df.segment_id)
    .groupBy("segment_name")
    .agg(F.avg("score").alias("avg_score"))
    .orderBy(F.desc("avg_score"))
)

print("Physical plan:")
complex_query.explain()

print("""
Reading explain output:
→ BroadcastHashJoin: small table broadcast ✅
→ SortMergeJoin: large table shuffle join ⚠️
→ HashAggregate: efficient aggregation ✅
→ Exchange: shuffle happening here ⚠️
→ Filter: predicate pushdown applied ✅
→ Scan: reading from source

Red flags in explain:
→ CartesianProduct: missing join condition!
→ SortMergeJoin on small table: add broadcast!
→ Exchange after Exchange: double shuffle!
""")

# 8. AQE statistics
print("\n=== AQE RUNTIME STATS ===")
print("""
With AQE enabled, Spark collects runtime stats:
→ Actual partition sizes after shuffle
→ Actual row counts per partition
→ Actual data distribution

AQE uses these to:
1. Coalesce small partitions (fewer tasks!)
2. Upgrade join strategy (SortMerge → Broadcast)
3. Handle skew (split large partitions!)

Monitor AQE in Spark UI:
→ SQL tab → query plan
→ Look for "AQE" annotations
→ See actual vs estimated statistics
""")

# Log to MLflow
mlflow.set_experiment("phase2_optimizations")
with mlflow.start_run(
        run_name="optimization_techniques"):
    mlflow.log_param("techniques_covered", [
        "single_scan", "avoid_udf",
        "multi_agg", "broadcast_tuning",
        "partition_opt", "caching", "aqe"
    ])
    mlflow.log_metric("optimizations", 7)
    print("\nOptimizations logged to MLflow!")
