# Phase 2 Day 5 — Advanced Partitioning
# Date: April 15, 2026
# Partitioning = Spark performance superpower!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.rdd import RDD
import mlflow

spark = SparkSession.builder \
    .appName("AdvancedPartitioning") \
    .getOrCreate()
sc = spark.sparkContext

print("="*60)
print("Advanced Spark Partitioning")
print("="*60)

"""
PARTITIONING MENTAL MODEL:

A partition = one chunk of data processed
              by one executor task

More partitions = more parallelism (up to cores)
Fewer partitions = less overhead

Rule of thumb:
→ 2-4 partitions per CPU core
→ 128MB-200MB per partition target size
→ 200 default shuffle partitions (tune me!)
"""

# 1. Understanding current partitioning
data = [(i, f"key_{i%10}", float(i))
        for i in range(100000)]
df = spark.createDataFrame(
    data, ["id", "key", "value"]
)
print(f"\nDefault partitions: "
      f"{df.rdd.getNumPartitions()}")

# 2. repartition vs coalesce
print("\n=== REPARTITION vs COALESCE ===")

# repartition — full shuffle, any number
df_8 = df.repartition(8)
print(f"After repartition(8): "
      f"{df_8.rdd.getNumPartitions()}")

# coalesce — no shuffle, reduce only
df_4 = df_8.coalesce(4)
print(f"After coalesce(4): "
      f"{df_4.rdd.getNumPartitions()}")

# repartition by column — co-locate same keys!
df_by_key = df.repartition(10, "key")
print(f"repartition by key: "
      f"{df_by_key.rdd.getNumPartitions()}")

# Verify all same-key rows in same partition
print("\nRows per partition after key repartition:")
df_by_key.withColumn(
    "partition_id",
    F.spark_partition_id()
).groupBy("partition_id", "key") \
 .count() \
 .orderBy("partition_id") \
 .show(20)

# 3. Custom Partitioner (RDD level)
print("\n=== CUSTOM PARTITIONER ===")

class ModelTypePartitioner:
    """
    Routes all rows of same model_type
    to same partition — enables faster
    subsequent groupBy on model_type!
    """
    def __init__(self, num_partitions):
        self.num_partitions = num_partitions

    def __call__(self, key):
        # Hash-based routing
        return hash(key) % self.num_partitions

# Apply custom partitioner
model_rdd = df.rdd.map(
    lambda row: (row.key, row)
)
partitioned_rdd = model_rdd.partitionBy(
    10,
    ModelTypePartitioner(10)
)
print(f"Custom partitioned: "
      f"{partitioned_rdd.getNumPartitions()}")

# Now groupByKey is FAST — all same keys together!
grouped = partitioned_rdd.groupByKey()
print(f"GroupByKey after partitioning: efficient!")

# 4. Bucket partitioning (Delta Lake style!)
print("\n=== BUCKETING IN SPARK SQL ===")

# Write bucketed table
(df.write
    .bucketBy(8, "key")  # 8 buckets on "key"
    .sortBy("key")
    .mode("overwrite")
    .saveAsTable("bucketed_data"))

# Read bucketed table
bucketed_df = spark.table("bucketed_data")
print(f"Bucketed table partitions: "
      f"{bucketed_df.rdd.getNumPartitions()}")

# Join two bucketed tables = NO SHUFFLE!
result = bucketed_df.join(
    bucketed_df.withColumnRenamed(
        "key", "key2"
    ).withColumnRenamed(
        "value", "value2"
    ),
    bucketed_df.key == F.col("key2")
)
result.explain()
print("Bucketed join avoids shuffle entirely!")

# 5. Partition pruning with Delta Lake
print("\n=== PARTITION PRUNING ===")
print("""
When you partition Delta table:
df.write.format("delta")
  .partitionBy("date", "model_type")
  .save("/tmp/partitioned_table")

Query with filter on partition column:
spark.read.format("delta")
  .load("/tmp/partitioned_table")
  .filter(F.col("date") == "2026-04-15")
  .filter(F.col("model_type") == "XGBoost")

Spark SKIPS all other partition folders!
→ date=2026-04-14/ ← SKIPPED
→ date=2026-04-15/model_type=XGBoost/ ← READ
→ date=2026-04-15/model_type=RF/ ← SKIPPED

Partition pruning can reduce I/O by 99%!
""")

# 6. Partition count tuning
print("\n=== PARTITION COUNT TUNING ===")
print("""
Rule 1: During shuffle
  Default: 200 partitions
  Tune:    spark.sql.shuffle.partitions
  Formula: Total data size / 128MB
  Example: 10GB data → 10000/128 ≈ 80 partitions

Rule 2: For reading
  Default: based on HDFS block size (128MB)
  Tune:    spark.sql.files.maxPartitionBytes
  AQE:     handles automatically!

Rule 3: After coalesce/repartition
  Target:  2-4x number of CPU cores
  4 workers × 4 cores = 16-32 partitions

AQE (Spark 3.2+) automates most of this!
""")

# 7. Log to MLflow
mlflow.set_experiment("phase2_partitioning")
with mlflow.start_run(run_name="partitioning_study"):
    mlflow.log_param("topic", "partitioning")
    mlflow.log_metric("concepts_covered", 6)
    print("\nPartitioning study logged to MLflow!")
