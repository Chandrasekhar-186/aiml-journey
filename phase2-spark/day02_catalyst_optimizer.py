# Phase 2 Day 2 — Catalyst Optimizer Internals
# Date: April 12, 2026
# Goal: Understand HOW Spark optimizes queries!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("CatalystDeepDive") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

print("="*60)
print("Catalyst Optimizer — 4 Phase Pipeline")
print("="*60)

"""
CATALYST OPTIMIZER PIPELINE:

SQL String / DataFrame API
        ↓
Phase 1: ANALYSIS (Unresolved Logical Plan)
  → Resolve column names from catalog
  → Resolve data types
  → Check table/view existence
  → Output: Resolved Logical Plan

Phase 2: LOGICAL OPTIMIZATION
  → Predicate Pushdown
    (move filters closer to data source)
  → Column Pruning
    (drop unused columns early)
  → Constant Folding
    (evaluate 1+1 → 2 at plan time)
  → Boolean Simplification
  → Output: Optimized Logical Plan

Phase 3: PHYSICAL PLANNING
  → Generate multiple physical plans
  → Choose best using cost model
  → Join strategy selection:
    BroadcastHashJoin (small table <10MB)
    SortMergeJoin (large tables)
    ShuffleHashJoin (medium tables)
  → Output: Physical Plan

Phase 4: CODE GENERATION (Tungsten)
  → Generate optimized JVM bytecode
  → Whole-stage code generation
  → Eliminates virtual function calls
  → SIMD vectorization where possible
  → Output: Compiled RDDs
"""

# 1. See Catalyst in action
data = [(i, f"model_{i%10}",
         float(i % 100), i % 5)
        for i in range(100000)]
df = spark.createDataFrame(
    data, ["id", "model", "score", "team"]
)

# Build complex query
query = (df
    .filter(F.col("score") > 80)        # predicate
    .select("model", "score", "team")    # column pruning
    .groupBy("team")
    .agg(F.avg("score").alias("avg"))
    .filter(F.col("avg") > 85)           # HAVING
    .orderBy(F.desc("avg"))
)

# 2. See all 4 optimizer phases
print("\n--- PARSED LOGICAL PLAN ---")
query.explain(mode="formatted")

print("\n--- EXTENDED (all phases) ---")
query.explain(extended=True)

# 3. Demonstrate predicate pushdown
print("\n=== PREDICATE PUSHDOWN DEMO ===")
# Catalyst automatically moves filter BEFORE join
# You write this:
result = (df
    .join(df.select("id", "model")
            .withColumnRenamed("id", "id2"),
          df.id == F.col("id2"))
    .filter(F.col("score") > 90)  # written AFTER join
)
# Catalyst rewrites to filter BEFORE join!
print("Filter after join (what you wrote):")
print("Catalyst moves filter before join automatically!")
result.explain()

# 4. Column pruning demo
print("\n=== COLUMN PRUNING DEMO ===")
pruned = (df
    .select("id", "model", "score",
             "team")  # select all 4
    .filter(F.col("score") > 80)
    .groupBy("team")
    .agg(F.avg("score"))
    # Catalyst knows "id" + "model" not needed
    # → drops them before reading!
)
pruned.explain()

# 5. Constant folding
print("\n=== CONSTANT FOLDING ===")
constant = df.filter(
    F.col("score") > (40 + 50)  # Catalyst → > 90
)
constant.explain()
print("Catalyst evaluates 40+50=90 at plan time!")

# 6. Join strategy selection
small_df = spark.createDataFrame(
    [(i, f"team_{i}") for i in range(5)],
    ["team_id", "team_name"]
)
large_join = df.join(small_df,
                      df.team == small_df.team_id)
print("\n=== JOIN STRATEGY ===")
large_join.explain()
print("Catalyst chooses BroadcastHashJoin")
print("because small_df < 10MB threshold!")

# 7. AQE — runtime optimization
print("\n=== ADAPTIVE QUERY EXECUTION ===")
print("""
AQE adds runtime optimization on top of Catalyst:

1. Dynamic partition coalescing:
   If shuffle produces many small partitions →
   AQE merges them automatically
   spark.sql.adaptive.coalescePartitions.enabled

2. Dynamic join conversion:
   At runtime, if build side < threshold →
   AQE converts SortMergeJoin → BroadcastHashJoin
   spark.sql.adaptive.localShuffleReader.enabled

3. Skew join optimization:
   Detects skewed partitions at runtime →
   Splits large partitions into smaller ones
   spark.sql.adaptive.skewJoin.enabled

All enabled by default in Spark 3.2+!
""")
