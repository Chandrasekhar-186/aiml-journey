# Phase 2 Day 7 — Delta Lake Internals
# Date: April 17, 2026
# Delta Lake = Databricks' most important product!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from pyspark.sql.types import *
import json
import mlflow

spark = SparkSession.builder \
    .appName("DeltaLakeInternals") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

print("="*60)
print("Delta Lake Internals — Complete Guide")
print("="*60)

"""
DELTA LAKE ARCHITECTURE:

Delta Table = Parquet files + _delta_log/

_delta_log/
├── 00000000000000000000.json  ← version 0
├── 00000000000000000001.json  ← version 1
├── 00000000000000000002.json  ← version 2
├── ...
└── 00000000000000000010.checkpoint.parquet
    ← checkpoint every 10 versions

Each .json file = one atomic commit containing:
→ add:      new files added
→ remove:   files logically deleted
→ metadata: schema changes
→ protocol: min reader/writer version
→ commitInfo: who, when, what operation

ACID via transaction log:
Atomicity:   entire commit or nothing
Consistency: schema validated on every write
Isolation:   optimistic concurrency control
Durability:  log persisted before data files
"""

# 1. Create Delta table and examine log
print("\n=== CREATING DELTA TABLE ===")
schema = StructType([
    StructField("id", IntegerType()),
    StructField("model", StringType()),
    StructField("accuracy", FloatType()),
    StructField("date", StringType())
])

data_v0 = [(1, "RF", 88.0, "2026-04-01"),
            (2, "XGB", 92.0, "2026-04-01"),
            (3, "NN", 95.0, "2026-04-01")]
df_v0 = spark.createDataFrame(data_v0, schema)

# Write version 0
df_v0.write.format("delta") \
     .mode("overwrite") \
     .save("/tmp/delta_internals")
print("Version 0 created!")

dt = DeltaTable.forPath(
    spark, "/tmp/delta_internals"
)

# 2. Multiple operations = multiple versions
print("\n=== CREATING MULTIPLE VERSIONS ===")

# Version 1: append
data_v1 = [(4, "CNN", 91.0, "2026-04-02")]
spark.createDataFrame(
    data_v1, schema
).write.format("delta") \
 .mode("append") \
 .save("/tmp/delta_internals")
print("Version 1: append")

# Version 2: update
dt.update(
    condition="model = 'RF'",
    set={"accuracy": "90.0"}
)
print("Version 2: update RF accuracy")

# Version 3: delete
dt.delete("accuracy < 90")
print("Version 3: delete low performers")

# Version 4: merge (upsert)
new_data = spark.createDataFrame([
    (2, "XGB", 94.5, "2026-04-03"),  # update
    (5, "LR", 85.0, "2026-04-03")    # insert
], schema)

dt.alias("old").merge(
    new_data.alias("new"),
    "old.id = new.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
print("Version 4: merge upsert")

# 3. View transaction history
print("\n=== TRANSACTION HISTORY ===")
dt.history().select(
    "version", "timestamp",
    "operation", "operationParameters"
).show(truncate=50)

# 4. Time travel — query any version!
print("\n=== TIME TRAVEL ===")
for v in range(5):
    try:
        count = spark.read.format("delta") \
            .option("versionAsOf", v) \
            .load("/tmp/delta_internals") \
            .count()
        print(f"Version {v}: {count} rows")
    except Exception as e:
        print(f"Version {v}: {str(e)[:40]}")

# Timestamp-based time travel
print("\nTimestamp-based time travel:")
print("""
spark.read.format("delta")
  .option("timestampAsOf", "2026-04-01")
  .load("/tmp/delta_internals")
""")

# 5. Schema evolution
print("\n=== SCHEMA EVOLUTION ===")
new_schema_data = spark.createDataFrame([
    (6, "BERT", 94.0, "2026-04-03",
     "transformer")  # new column!
], ["id", "model", "accuracy",
    "date", "architecture"])

# Without mergeSchema → error!
# With mergeSchema → evolves schema!
new_schema_data.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("/tmp/delta_internals")
print("Schema evolved — added 'architecture' col!")

current = spark.read.format("delta") \
    .load("/tmp/delta_internals")
print(f"New schema: {current.schema.simpleString()}")
current.show()

# 6. OPTIMIZE + ZORDER
print("\n=== OPTIMIZE + Z-ORDER ===")
spark.sql("""
    OPTIMIZE delta.`/tmp/delta_internals`
    ZORDER BY (model)
""")
print("Table optimized with Z-Order on model!")
print("""
OPTIMIZE: compacts many small files into larger ones
          → Solves "small file problem"
          → Faster reads (fewer files to open)

ZORDER BY (col):
          → Co-locates rows with same col value
          → Skips files that don't contain query value
          → Dramatically speeds up:
            WHERE model = 'XGBoost'
            WHERE date BETWEEN x AND y
""")

# 7. VACUUM — cleanup old files
print("\n=== VACUUM ===")
print("""
VACUUM delta.`/tmp/delta_internals`
      RETAIN 168 HOURS  -- 7 days default

What it does:
→ Removes data files no longer in latest version
→ Files older than retention period = deleted
→ CANNOT time travel past vacuum point!

Best practice:
→ Keep 7 days minimum retention
→ Run weekly in production
→ Never vacuum < 7 days (breaks concurrent reads!)
""")

# 8. Delta Lake stats
print("\n=== DELTA TABLE STATS ===")
detail = dt.detail()
detail.select(
    "format", "numFiles",
    "sizeInBytes", "partitionColumns"
).show()

# 9. Log to MLflow
mlflow.set_experiment("phase2_delta_lake")
with mlflow.start_run(
        run_name="delta_internals"):
    mlflow.log_param(
        "topic", "delta_lake_internals"
    )
    mlflow.log_metric(
        "versions_created", 5
    )
    print("\nDelta Lake internals logged!")
