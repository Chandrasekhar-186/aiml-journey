# Phase 1 Capstone — Day 2
# PySpark + Delta Lake Integration
# Date: April 3, 2026

import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pandas as pd

print("="*55)
print("Capstone Day 2: PySpark + Delta Lake")
print("="*55)

# 1. Load experiment data into Spark
pandas_df = pd.read_csv('experiments_dataset.csv')
spark_df = spark.createDataFrame(pandas_df)

print(f"\nLoaded {spark_df.count()} experiments")
spark_df.printSchema()

# 2. Data quality checks
print("\nData Quality Report:")
print(f"  Null counts:")
spark_df.select([
    F.sum(F.col(c).isNull().cast("int"))
     .alias(f"null_{c}")
    for c in spark_df.columns
]).show()

# 3. Feature engineering with PySpark
enriched_df = (spark_df
    .withColumn("accuracy_bucket",
        F.when(F.col("accuracy") >= 0.95, "S-tier")
         .when(F.col("accuracy") >= 0.90, "A-tier")
         .when(F.col("accuracy") >= 0.85, "B-tier")
         .otherwise("C-tier")
    )
    .withColumn("efficiency_score",
        F.round(
            F.col("accuracy") /
            F.col("train_time"), 4
        )
    )
    .withColumn("rank_by_accuracy",
        F.rank().over(
            Window.partitionBy("model_type")
            .orderBy(F.desc("accuracy"))
        )
    )
    .withColumn("ingested_at",
        F.current_timestamp()
    )
)

print("\nEnriched DataFrame:")
enriched_df.select(
    "exp_id", "model_type", "accuracy",
    "accuracy_bucket", "efficiency_score",
    "rank_by_accuracy"
).show(10)

# 4. Analytics
print("\nModel Performance Summary:")
(enriched_df
    .groupBy("model_type", "accuracy_bucket")
    .agg(
        F.count("*").alias("count"),
        F.round(F.avg("accuracy"), 4)
         .alias("avg_accuracy"),
        F.round(F.avg("efficiency_score"), 4)
         .alias("avg_efficiency")
    )
    .orderBy("model_type", "accuracy_bucket")
    .show()
)

# 5. Write to Delta Lake!
print("\nWriting to Delta Lake...")
(enriched_df
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("model_type")
    .save("/tmp/capstone_experiments")
)
print("Delta Lake table created!")

# 6. Read back + verify
delta_df = spark.read.format("delta").load(
    "/tmp/capstone_experiments"
)
print(f"Delta table rows: {delta_df.count()}")
print(f"Partitions: {delta_df.rdd.getNumPartitions()}")

# 7. Delta operations
dt = DeltaTable.forPath(
    spark, "/tmp/capstone_experiments"
)

# Update low performers
dt.update(
    condition="accuracy < 0.6",
    set={"accuracy_bucket": "'D-tier'"}
)

# Show table history
print("\nDelta table history:")
dt.history().select(
    "version", "operation", "timestamp"
).show()

# Time travel!
print("\nOriginal data (version 0):")
spark.read.format("delta").option(
    "versionAsOf", 0
).load("/tmp/capstone_experiments").groupBy(
    "model_type"
).count().show()

# 8. Log to MLflow
mlflow.set_experiment("capstone_ml_analyzer")
with mlflow.start_run(run_name="spark_delta_integration"):
    mlflow.log_param("storage", "Delta Lake")
    mlflow.log_param("partitioned_by", "model_type")
    mlflow.log_metric("total_rows",
                       enriched_df.count())
    mlflow.log_metric("delta_versions",
                       dt.history().count())
    print("\nCapstone Day 2 logged to MLflow!")

print("\nCapstone Day 2 Complete!")
print("Tomorrow: Meta-model training!")
```

---

### 🎯 MOCK INTERVIEW — Coding Round (45 min)

**This is your first real mock interview! Timer starts NOW!**

I'll give you 2 problems. You have **45 minutes total**. Solve both, explain your approach out loud (type it here), then I'll score you.

**Rules:**
- No looking up solutions
- Write your approach in comments first
- Then code it
- Aim for optimal solution

---

**Problem 1 (20 min):** [LeetCode #271 equivalent]
```
Given a list of strings words and a string s,
return true if s can be formed by concatenating
words from the list (words can be reused).

Example:
words = ["cat", "cats", "and", "sand", "dog"]
s = "catsanddog" → True
s = "catsandog"  → False
