# Phase 2 Day 8 — Delta Lake Advanced
# Date: April 18, 2026
# Liquid Clustering + Delta Sharing + Unity Catalog

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable
import mlflow

spark = SparkSession.builder \
    .appName("DeltaAdvanced") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog"
            ".DeltaCatalog") \
    .getOrCreate()

print("="*60)
print("Delta Lake Advanced Features")
print("="*60)

# 1. Liquid Clustering vs Z-Ordering
print("\n=== LIQUID CLUSTERING ===")
print("""
Traditional Z-ORDER problems:
→ Must rewrite entire table to re-cluster
→ Expensive for frequently updated tables
→ Cannot change cluster columns easily
→ Requires OPTIMIZE command manually

Liquid Clustering (Delta 3.0+ / DBR 13.0+):
→ Clusters data INCREMENTALLY
→ Can change cluster columns anytime
→ Automatic clustering in background
→ Better performance for mixed workloads

Syntax:
CREATE TABLE ml_experiments
CLUSTER BY (model_type, date)

Or for existing table:
ALTER TABLE ml_experiments
CLUSTER BY (model_type, date)

Then: OPTIMIZE ml_experiments
      (uses liquid clustering automatically!)

When to use Liquid Clustering:
✅ High-cardinality columns (user_id, event_id)
✅ Columns you filter on frequently
✅ Tables with frequent inserts/updates
✅ When partition columns don't match filters

When to use traditional partitioning:
✅ Low-cardinality (date, region, model_type)
✅ Most queries filter on this column
✅ Predictable data distribution
""")

# 2. Demonstrate partitioning vs clustering
data = [(i, f"model_{i%10}",
         f"2026-{(i%12)+1:02d}-01",
         float(i%100))
        for i in range(10000)]
df = spark.createDataFrame(
    data, ["id", "model_type", "date", "score"]
)

# Traditional partitioning
df.write.format("delta") \
    .partitionBy("model_type") \
    .mode("overwrite") \
    .save("/tmp/partitioned_delta")
print("Partitioned table written!")

# Check partition structure
import subprocess
print("\nPartitioned table structure:")
print("""
/tmp/partitioned_delta/
├── _delta_log/
├── model_type=CNN/
│   └── part-0001.parquet
├── model_type=RF/
│   └── part-0002.parquet
└── model_type=XGBoost/
    └── part-0003.parquet
""")

# 3. Deletion Vectors (Delta 2.3+)
print("\n=== DELETION VECTORS ===")
print("""
Traditional DELETE in Delta Lake:
→ Copy-on-write: rewrite entire file
→ Expensive for large files with few deletes

Deletion Vectors:
→ Store deleted row positions in separate file
→ No file rewrite needed!
→ Much faster for small deletes
→ Applied at read time (merge-on-read)

Enable:
ALTER TABLE ml_experiments
SET TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true'
)

Performance comparison:
Traditional: DELETE 100 rows from 1M row file
             → Rewrite entire 1M rows
Deletion Vec: DELETE 100 rows
              → Write tiny 100-row bitmap
              → 1000× faster!
""")

# 4. Delta Sharing
print("\n=== DELTA SHARING ===")
print("""
Delta Sharing = open protocol for sharing
live Delta tables across organizations

Without Delta Sharing:
→ Export CSV → send via email → stale data!
→ Copy data to recipient's storage → expensive

With Delta Sharing:
→ Share LIVE table access (read-only)
→ No data copying!
→ Cross-cloud, cross-platform
→ Works with Spark, pandas, PowerBI, Excel

Architecture:
Data Provider          Data Recipient
    ↓                       ↓
Delta Table ←── Delta Sharing Server
                    ↓
            Sharing Protocol (REST API)
                    ↓
            Recipient's Spark/pandas

Use case at Databricks:
→ Share ML experiment results with partners
→ Share model metrics across organizations
→ Cross-cloud data collaboration
""")

# 5. Change Data Feed (CDF)
print("\n=== CHANGE DATA FEED ===")

# Enable CDF on table
spark.sql("""
    CREATE TABLE IF NOT EXISTS ml_models_cdf
    (id INT, model STRING, accuracy FLOAT)
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
""")

# Insert some data
spark.sql("""
    INSERT INTO ml_models_cdf VALUES
    (1, 'RF', 88.0),
    (2, 'XGB', 92.0),
    (3, 'NN', 95.0)
""")

# Update some rows
spark.sql("""
    UPDATE ml_models_cdf
    SET accuracy = 90.0
    WHERE model = 'RF'
""")

# Read the changes!
changes = spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", 0) \
    .table("ml_models_cdf")

print("\nChange Data Feed output:")
changes.show()
print("""
_change_type column:
→ insert:         new rows
→ update_preimage:  row BEFORE update
→ update_postimage: row AFTER update
→ delete:         deleted rows

Use cases:
→ Stream changes to downstream systems
→ Audit trail for compliance
→ Incremental ML feature computation
→ Real-time dashboards
""")

# 6. Unity Catalog overview
print("\n=== UNITY CATALOG ===")
print("""
Unity Catalog = Databricks' data governance layer

3-level namespace:
catalog.schema.table

Example:
ml_catalog.experiments.results
    ↑           ↑          ↑
  catalog     schema     table

Features:
→ Fine-grained access control (row + column level!)
→ Automated data lineage tracking
→ Built-in search and discovery
→ Cross-workspace data sharing
→ AI-powered data classification

In SQL:
USE CATALOG ml_catalog;
USE SCHEMA experim
