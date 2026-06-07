# Phase 4 Day 3 — Databricks Workflows + DLT
# Date: June 14, 2026
# Production pipeline orchestration!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow
import json

spark = SparkSession.builder \
    .appName("WorkflowsDLT") \
    .getOrCreate()

print("="*60)
print("Databricks Workflows + Delta Live Tables")
print("="*60)

"""
DATABRICKS WORKFLOWS — COMPLETE GUIDE

What is it?
Databricks-native job orchestration
→ Schedule notebooks, Python scripts, JARs
→ Chain tasks with dependencies
→ Handle failures + retries
→ Monitor + alert on failures
→ No external orchestrator needed!

vs Airflow:
Airflow:    external, general purpose
            complex setup, any DAG
Workflows:  Databricks-native, simpler
            tight integration with Spark/ML
            built-in Git integration

ANATOMY OF A WORKFLOW:

Job:          the overall pipeline
Task:         one unit of work (notebook/script)
Cluster:      compute for each task
Dependencies: task A → task B → task C
Schedule:     cron or triggered

Example ML pipeline:
Task 1: data_ingestion (every hour)
    ↓
Task 2: feature_engineering
    ↓
Task 3: model_training (if new data)
    ↓
Task 4: model_evaluation
    ↓
Task 5: model_promotion (if accuracy > 0.9)
    ↓
Task 6: monitoring_update

Job clusters (new cluster per run):
→ Clean environment each time
→ Auto-terminate after job
→ More expensive but reliable

Interactive clusters:
→ Shared across runs
→ Faster startup
→ Risk of state contamination

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DELTA LIVE TABLES (DLT) — DECLARATIVE ETL

Standard ETL:  write code for EACH step
               manage dependencies manually
               handle failures yourself

DLT:           declare WHAT you want
               DLT handles HOW + WHEN
               automatic lineage + quality

DLT Pipeline:
@dlt.table
def bronze_data():
    return spark.read.format("json")\
                .load("/raw/data")

@dlt.table
@dlt.expect("valid_id", "id IS NOT NULL")
def silver_data():
    return dlt.read("bronze_data")\
              .filter("value > 0")\
              .withColumn("processed_at",
                           current_timestamp())

@dlt.table
def gold_aggregates():
    return dlt.read("silver_data")\
              .groupBy("category")\
              .agg(avg("value"))

DLT Quality Constraints:
@dlt.expect:          log failures, continue
@dlt.expect_or_drop:  drop bad rows
@dlt.expect_or_fail:  stop pipeline on failure

DLT Modes:
Triggered:   run once on schedule (batch)
Continuous:  always running (streaming)

DLT vs manual ETL:
Manual: write DAG, manage dependencies,
        handle checkpoints, manage schema
DLT:    declare tables, DLT handles rest!
        Built-in data quality + lineage
        Event-driven incremental processing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTOLOADER — INCREMENTAL FILE INGESTION

Problem: new files arrive continuously
Standard: reprocess ALL files each run (slow!)

Autoloader solution:
→ Detect new files automatically
→ Process ONLY new files
→ Exactly-once semantics
→ Scales to millions of files!

spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "json") \
    .load("/raw/landing/") \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "/ckpt/") \
    .start("/bronze/")

cloudFiles = Autoloader!
→ Tracks which files already processed
→ New file arrives → picked up automatically
→ No need to track state manually
"""

# 1. Simulate Workflow tasks
print("\n=== SIMULATING WORKFLOW TASKS ===")

def task_data_ingestion():
    """Task 1: Ingest raw data"""
    print("Task 1: Data Ingestion")
    # Simulate incoming data
    data = [
        (i, float(i % 100) / 100,
         f"cat_{i % 5}", i % 2,
         "2026-06-14")
        for i in range(500)
    ]
    df = spark.createDataFrame(
        data,
        ["id", "value", "category",
         "label", "date"]
    )
    df.write.format("delta") \
        .mode("overwrite") \
        .save("/tmp/workflow/bronze")
    count = df.count()
    print(f"  ✅ Ingested {count} records")
    return count

def task_feature_engineering():
    """Task 2: Compute features"""
    print("Task 2: Feature Engineering")
    df = spark.read.format("delta") \
        .load("/tmp/workflow/bronze")

    features_df = df \
        .withColumn("value_squared",
                     F.col("value") ** 2) \
        .withColumn("value_log",
                     F.log(F.col("value") + 1)) \
        .withColumn("is_high_value",
                     (F.col("value") > 0.7)
                     .cast("int")) \
        .withColumn("feature_ts",
                     F.current_timestamp())

    features_df.write.format("delta") \
        .mode("overwrite") \
        .save("/tmp/workflow/silver")
    print(f"  ✅ Features computed: "
          f"{features_df.count()} rows")

def task_model_training():
    """Task 3: Train model on new features"""
    print("Task 3: Model Training")
    import pandas as pd
    from sklearn.ensemble import \
        GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    df = spark.read.format("delta") \
        .load("/tmp/workflow/silver") \
        .toPandas()

    features = ["value", "value_squared",
                 "value_log", "is_high_value"]
    X = df[features].values
    y = df["label"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mlflow.set_experiment(
        "phase4_workflow_training"
    )
    with mlflow.start_run(
            run_name="workflow_automated_train"):
        model = GradientBoostingClassifier(
            n_estimators=50, random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(
            y_test, model.predict(X_test)
        )
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param(
            "triggered_by", "workflow"
        )
        mlflow.sklearn.log_model(
            model, "workflow_model"
        )
        print(f"  ✅ Model trained: "
              f"accuracy={acc:.4f}")
        return acc

def task_model_evaluation(accuracy: float):
    """Task 4: Decide whether to promote"""
    print("Task 4: Model Evaluation")
    threshold = 0.80
    promote = accuracy >= threshold
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Threshold: {threshold}")
    print(f"  Decision: "
          f"{'✅ PROMOTE' if promote else '❌ REJECT'}")
    return promote

def task_monitoring_update():
    """Task 5: Update monitoring stats"""
    print("Task 5: Monitoring Update")
    gold_df = spark.read.format("delta") \
        .load("/tmp/workflow/silver") \
        .groupBy("category") \
        .agg(
            F.count("*").alias("count"),
            F.avg("value").alias("avg_value"),
            F.stddev("value").alias("std_value")
        )
    gold_df.write.format("delta") \
        .mode("overwrite") \
        .save("/tmp/workflow/gold")
    print(f"  ✅ Gold layer updated")
    gold_df.show()

# Run the full workflow!
print("\n" + "="*40)
print("RUNNING ML WORKFLOW PIPELINE")
print("="*40)

count = task_data_ingestion()
task_feature_engineering()
accuracy = task_model_training()
promote = task_model_evaluation(accuracy)
task_monitoring_update()

print("\n✅ WORKFLOW COMPLETE!")

# 2. Simulate DLT declarations
print("\n=== DELTA LIVE TABLES (DLT) ===")
print("""
# In production on Databricks:
import dlt
from pyspark.sql.functions import *

# Bronze: raw ingestion
@dlt.table(
    comment="Raw ML training data",
    table_properties={"quality": "bronze"}
)
def bronze_ml_data():
    return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load("/raw/ml_data/"))

# Silver: cleaned + validated
@dlt.table(
    comment="Validated ML features",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop(
    "valid_value", "value IS NOT NULL"
)
@dlt.expect_or_drop(
    "positive_value", "value > 0"
)
def silver_ml_features():
    return (dlt.read_stream("bronze_ml_data")
        .withColumn("value_log",
                     log(col("value") + 1))
        .withColumn("processed_at",
                     current_timestamp()))

# Gold: aggregated metrics
@dlt.table(
    comment="ML monitoring metrics",
    table_properties={"quality": "gold"}
)
def gold_ml_metrics():
    return (dlt.read("silver_ml_features")
        .groupBy("category", "date")
        .agg(
            count("*").alias("count"),
            avg("value").alias("avg_value"),
            stddev("value").alias("std_val")
        ))

# DLT handles:
# → Dependency resolution (bronze→silver→gold)
# → Schema evolution automatically
# → Data quality enforcement
# → Incremental updates (only new data!)
# → Lineage tracking in Unity Catalog
""")

# 3. Autoloader simulation
print("\n=== AUTOLOADER PATTERN ===")
print("""
# Autoloader: detect + process new files
# cloudFiles = Databricks Autoloader magic!

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation",
            "/schema/checkpoint/")
    .load("/landing/raw_data/")

    # Transformations
    .withColumn("ingested_at",
                 current_timestamp())
    .withColumn("source_file",
                 input_file_name())

    .writeStream
    .format("delta")
    .option("checkpointLocation",
            "/checkpoints/bronze/")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)  # batch!
    # OR: trigger(processingTime="1 minute")
    .start("/bronze/ml_data/")
)

# New file lands in /landing/raw_data/
# → Autoloader detects immediately
# → Processes ONLY that new file
# → Writes to Delta with exactly-once!
# → Checkpoint tracks progress
""")

# 4. Workflow JSON config
print("\n=== WORKFLOW JSON CONFIG ===")
workflow_config = {
    "name": "ML_Training_Pipeline",
    "schedule": {
        "quartz_cron_expression": "0 0 2 * * ?",
        "timezone_id": "UTC"
    },
    "tasks": [
        {
            "task_key": "data_ingestion",
            "notebook_task": {
                "notebook_path":
                    "/Workflows/ingestion"
            },
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 4
            }
        },
        {
            "task_key": "feature_engineering",
            "depends_on": [
                {"task_key": "data_ingestion"}
            ],
            "notebook_task": {
                "notebook_path":
                    "/Workflows/features"
            }
        },
        {
            "task_key": "model_training",
            "depends_on": [
                {"task_key":
                     "feature_engineering"}
            ],
            "python_wheel_task": {
                "package_name": "ml_pipeline",
                "entry_point": "train"
            }
        },
        {
            "task_key": "model_promotion",
            "depends_on": [
                {"task_key": "model_training"}
            ],
            "condition_task": {
                "op": "GREATER_THAN",
                "left": "{{tasks.model_training"
                         ".values.accuracy}}",
                "right": "0.9"
            }
        }
    ],
    "email_notifications": {
        "on_failure": ["chandra@example.com"],
        "on_success": ["chandra@example.com"]
    }
}

print(json.dumps(workflow_config, indent=2))

mlflow.set_experiment("phase4_workflows")
with mlflow.start_run(
        run_name="workflow_orchestration"):
    mlflow.log_params({
        "orchestration": "Databricks_Workflows",
        "etl": "Delta_Live_Tables",
        "ingestion": "Autoloader",
        "schedule": "daily_2am_UTC",
        "tasks": len(
            workflow_config["tasks"]
        )
    })
    mlflow.log_metric(
        "workflow_tasks",
        len(workflow_config["tasks"])
    )
    print("\nWorkflow config logged!")

print("\n" + "="*60)
print("Workflows + DLT + Autoloader — MASTERED!")
print("Phase 4 Day 3 COMPLETE! 🏭")
print("="*60)
