# Phase 2 Day 4 — RDD Operations
# Date: April 14, 2026
# REQUIRED for Spark cert — 15% of exam!

from pyspark.sql import SparkSession
from pyspark import SparkContext
import mlflow

spark = SparkSession.builder \
    .appName("RDDOperations") \
    .getOrCreate()
sc = spark.sparkContext

print("="*60)
print("RDD Operations — Complete Guide")
print("="*60)

# 1. Creating RDDs
print("\n=== CREATING RDDs ===")

# From collection
rdd1 = sc.parallelize([1, 2, 3, 4, 5],
                        numSlices=3)
print(f"From collection: {rdd1.collect()}")
print(f"Partitions: {rdd1.getNumPartitions()}")

# From file
# rdd2 = sc.textFile("path/to/file.txt")

# From DataFrame
df = spark.createDataFrame(
    [(1, "spark"), (2, "mlflow"),
     (3, "delta")],
    ["id", "tool"]
)
rdd_from_df = df.rdd
print(f"From DataFrame: {rdd_from_df.collect()}")

# 2. Transformations (lazy!)
print("\n=== RDD TRANSFORMATIONS ===")
numbers = sc.parallelize(range(1, 11))

# map — apply function to each element
squared = numbers.map(lambda x: x ** 2)
print(f"map (squared): {squared.collect()}")

# filter — keep matching elements
evens = numbers.filter(lambda x: x % 2 == 0)
print(f"filter (evens): {evens.collect()}")

# flatMap — map + flatten
words = sc.parallelize(
    ["spark is fast", "mlflow is great"]
)
word_list = words.flatMap(
    lambda line: line.split(" ")
)
print(f"flatMap: {word_list.collect()}")

# mapPartitions — function per partition
def process_partition(iterator):
    # More efficient than map for DB connections!
    yield sum(iterator)

partition_sums = numbers.mapPartitions(
    process_partition
)
print(f"mapPartitions: {partition_sums.collect()}")

# mapPartitionsWithIndex
def with_index(idx, iterator):
    yield f"Partition {idx}: {list(iterator)}"

indexed = numbers.mapPartitionsWithIndex(with_index)
for item in indexed.collect():
    print(f"  {item}")

# 3. Key-Value RDD operations
print("\n=== KEY-VALUE RDD OPERATIONS ===")
pairs = sc.parallelize([
    ("spark", 1), ("mlflow", 1),
    ("spark", 1), ("delta", 1),
    ("mlflow", 1), ("spark", 1)
])

# reduceByKey — combine values for same key
counts = pairs.reduceByKey(lambda a, b: a + b)
print(f"reduceByKey: {counts.collect()}")

# groupByKey — group values (use reduceByKey instead!)
grouped = pairs.groupByKey()
print(f"groupByKey: "
      f"{[(k, list(v)) for k, v in grouped.collect()]}")

# IMPORTANT: reduceByKey >> groupByKey!
# reduceByKey: combines on map side first (less shuffle)
# groupByKey: shuffles ALL values then groups (slow!)

# sortByKey
sorted_rdd = counts.sortByKey()
print(f"sortByKey: {sorted_rdd.collect()}")

# mapValues — transform values only
upper_vals = counts.mapValues(
    lambda v: v * 10
)
print(f"mapValues: {upper_vals.collect()}")

# join — like SQL join
rdd_a = sc.parallelize(
    [("spark", "fast"), ("mlflow", "tracking")]
)
rdd_b = sc.parallelize(
    [("spark", 2015), ("mlflow", 2018)]
)
joined = rdd_a.join(rdd_b)
print(f"join: {joined.collect()}")

# leftOuterJoin
left_join = rdd_a.leftOuterJoin(rdd_b)
print(f"leftOuterJoin: {left_join.collect()}")

# 4. Actions (trigger execution)
print("\n=== RDD ACTIONS ===")
data = sc.parallelize([3, 1, 4, 1, 5, 9, 2, 6])

print(f"count():    {data.count()}")
print(f"first():    {data.first()}")
print(f"take(3):    {data.take(3)}")
print(f"top(3):     {data.top(3)}")
print(f"min():      {data.min()}")
print(f"max():      {data.max()}")
print(f"sum():      {data.sum()}")
print(f"mean():     {data.mean()}")
print(f"collect():  {data.collect()}")

# reduce — aggregate entire RDD
total = data.reduce(lambda a, b: a + b)
print(f"reduce(sum): {total}")

# fold — like reduce with zero value
folded = data.fold(0, lambda a, b: a + b)
print(f"fold(sum):   {folded}")

# aggregate — most powerful action!
# (zeroValue, seqOp, combOp)
stats = data.aggregate(
    (0, 0),  # (sum, count)
    lambda acc, val: (acc[0]+val, acc[1]+1),
    lambda a, b: (a[0]+b[0], a[1]+b[1])
)
mean = stats[0] / stats[1]
print(f"aggregate mean: {mean:.2f}")

# 5. Set operations
print("\n=== SET OPERATIONS ===")
rdd1 = sc.parallelize([1, 2, 3, 4, 5])
rdd2 = sc.parallelize([3, 4, 5, 6, 7])

print(f"union:        {rdd1.union(rdd2).collect()}")
print(f"intersection: "
      f"{rdd1.intersection(rdd2).collect()}")
print(f"subtract:     "
      f"{rdd1.subtract(rdd2).collect()}")

# 6. Persistence
print("\n=== RDD PERSISTENCE ===")
expensive_rdd = numbers.map(
    lambda x: x ** 2
).filter(
    lambda x: x > 10
)
expensive_rdd.cache()
print(f"First action:  {expensive_rdd.collect()}")
print(f"Second action: {expensive_rdd.sum()}")
# Second action uses cached data!
expensive_rdd.unpersist()

# 7. DataFrame vs RDD — when to use which
print("\n=== DATAFRAME vs RDD ===")
print("""
DataFrame API (ALWAYS prefer this!):
✅ Catalyst optimization
✅ Tungsten execution
✅ Simpler API
✅ Better performance
✅ Schema enforcement
Use for: ALL data processing tasks

RDD API (only when necessary):
⚠️ No Catalyst optimization
⚠️ No Tungsten
⚠️ More verbose
Use for:
→ Unstructured data (log parsing)
→ Custom partitioning logic
→ When Spark cert requires it 😄
→ Low-level operations unavailable in DF
""")

# 8. Log to MLflow
mlflow.set_experiment("phase2_rdd_ops")
with mlflow.start_run(run_name="RDD_mastery"):
    mlflow.log_param("topic", "RDD_operations")
    mlflow.log_metric("rdd_ops_covered", 15)
    print("\nRDD operations logged to MLflow!")
