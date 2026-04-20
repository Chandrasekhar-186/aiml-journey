# Phase 2 Day 4 — Data Skew + Salting
# Date: April 14, 2026

from pyspark.sql import functions as F
import random

spark = SparkSession.builder \
    .appName("DataSkew").getOrCreate()

print("="*55)
print("Data Skew — The #1 Spark Problem")
print("="*55)

# 1. Simulate skewed data
print("\nCreating skewed dataset...")
# 90% of data has same key = SKEW!
skewed_data = (
    [("hot_key", i) for i in range(90000)] +
    [("cold_key_1", i) for i in range(5000)] +
    [("cold_key_2", i) for i in range(5000)]
)
random.shuffle(skewed_data)
skewed_df = spark.createDataFrame(
    skewed_data, ["key", "value"]
)

print(f"Total rows: {skewed_df.count()}")
print("\nKey distribution (SKEWED!):")
skewed_df.groupBy("key").count() \
         .orderBy(F.desc("count")).show()

# 2. Skew causes slow tasks in Spark UI
# hot_key partition = 90x larger than others!
# One task takes 90x longer → bottleneck!

# 3. Fix: Salting technique
print("\n=== SALTING FIX ===")
SALT_FACTOR = 10  # split hot key into 10 parts

# Add random salt to key
salted_df = skewed_df.withColumn(
    "salted_key",
    F.concat(
        F.col("key"),
        F.lit("_"),
        (F.rand() * SALT_FACTOR).cast("int")
    )
)
print("After salting — key distribution:")
salted_df.groupBy("salted_key") \
         .count() \
         .orderBy(F.desc("count")) \
         .show(15)

# 4. Salted groupBy
result = (salted_df
    .groupBy("salted_key")
    .agg(F.sum("value").alias("partial_sum"))
    .withColumn("original_key",
        F.split(F.col("salted_key"), "_")[0]
    )
    .groupBy("original_key")
    .agg(F.sum("partial_sum").alias("total"))
)
print("Final aggregation (after de-salting):")
result.show()

# 5. AQE automatic skew handling
print("\n=== AQE SKEW JOIN ===")
print("""
With AQE enabled (Spark 3.2+):
spark.conf.set(
    "spark.sql.adaptive.skewJoin.enabled",
    "true"
)
spark.conf.set(
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor",
    "5"  # partition is skewed if 5x median size
)

AQE automatically:
1. Detects skewed partitions at runtime
2. Splits them into smaller sub-partitions
3. Replicates the other side to match
→ No manual salting needed!
→ But understanding salting shows depth!
""")
