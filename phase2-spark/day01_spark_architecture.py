# Phase 2 — Day 1: Spark Architecture Deep Dive
# Date: April 11, 2026
# Goal: Understand Spark like its creators do!

"""
SPARK ARCHITECTURE — COMPLETE MENTAL MODEL

Driver Program
│
├── SparkContext / SparkSession
│   └── Creates logical execution plan
│
├── DAGScheduler
│   └── Converts logical plan → stages
│   └── Identifies shuffle boundaries
│
├── TaskScheduler
│   └── Assigns tasks to executors
│   └── Handles failures + retries
│
└── Cluster Manager (YARN/K8s/Standalone)
    └── Manages resources
    └── Launches executors

Executors (Workers)
├── Execute tasks in parallel
├── Store cached data (RDD/DataFrame)
└── Return results to driver

KEY INSIGHT: Driver is the brain, Executors are muscle
Never collect() large DataFrames to driver!
"""

# Let's verify understanding with code
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Phase2Day1") \
    .config("spark.sql.adaptive.enabled",
            "true") \
    .config("spark.sql.shuffle.partitions",
            "auto") \
    .getOrCreate()

# 1. Understanding jobs, stages, tasks
print("=== Spark Execution Hierarchy ===")
print("""
Job:   Triggered by ONE action (show, count, write)
Stage: Group of tasks with NO shuffle between them
Task:  ONE partition processed by ONE executor core

Example with groupBy:
df.filter().groupBy().agg().show()
         ↑              ↑         ↑
    Stage 1 start  Shuffle!  Stage 2 + Action
         (map side)           (reduce side)
""")

# 2. Demonstrate lazy evaluation
data = [(i, f"model_{i%5}", float(i%100))
        for i in range(10000)]
df = spark.createDataFrame(
    data, ["id", "model", "score"]
)

# ALL of these are lazy — nothing runs yet!
filtered = df.filter(F.col("score") > 50)
grouped = filtered.groupBy("model").agg(
    F.avg("score").alias("avg_score"),
    F.count("*").alias("count")
)
ranked = grouped.orderBy(F.desc("avg_score"))

print("Transformations defined — nothing ran yet!")
print("Execution plan (before running):")
ranked.explain()  # shows logical plan

# NOW it executes (action triggered!)
print("\nActual results (action triggers execution):")
ranked.show()

# 3. Understand partitioning
print(f"\nDefault partitions: "
      f"{df.rdd.getNumPartitions()}")
print(f"After groupBy (shuffle): "
      f"{ranked.rdd.getNumPartitions()}")

# 4. Stage boundaries
print("\nOperations that create stage boundaries:")
print("(i.e. cause a shuffle):")
boundaries = [
    "groupBy() + agg()",
    "join() — unless broadcast",
    "distinct()",
    "repartition()",
    "orderBy() / sort()",
    "union() — sometimes"
]
for op in boundaries:
    print(f"  → {op}")

# 5. AQE — Adaptive Query Execution
print("\nAQE — Spark 3.0+ game changer:")
print("""
Without AQE: plan fixed before execution
With AQE:    plan changes DURING execution based on:
  → Actual partition sizes (fixes skew!)
  → Actual row counts (changes join strategy!)
  → Runtime statistics

AQE features:
  → Coalesces small shuffle partitions
  → Converts sort-merge join to broadcast join
  → Skew join optimization
""")
