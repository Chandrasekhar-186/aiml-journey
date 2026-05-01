# Phase 2 Day 17 — Advanced Streaming Patterns
# Date: April 27, 2026

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow

spark = SparkSession.builder \
    .appName("AdvancedStreaming") \
    .getOrCreate()

print("="*60)
print("Advanced Streaming Patterns")
print("="*60)

# 1. Stream-static join
print("\n=== STREAM-STATIC JOIN ===")
print("""
Pattern: Enrich streaming data with static lookup

Use case: join live predictions with
          model metadata from Delta Lake

streaming_preds.join(
    static_model_metadata,  # batch DataFrame!
    "model_id"
)

Rules:
→ Static side loaded once at query start
→ Refreshed when query restarts
→ No shuffle needed for static join
→ Best for: lookup tables, config data

For frequently changing data:
→ Use Delta table + periodic refresh
→ Or use broadcast variable
""")

# Create static lookup
model_metadata = spark.createDataFrame([
    ("RF", "sklearn", "v1.2", "tabular"),
    ("XGB", "xgboost", "v0.9", "tabular"),
    ("NN", "pytorch", "v2.0", "image"),
], ["model_id", "framework",
    "version", "data_type"])

# Simulated streaming
rate_stream = (spark.readStream
    .format("rate")
    .option("rowsPerSecond", 10)
    .load()
    .withColumn("model_id",
        F.element_at(
            F.array(F.lit("RF"),
                     F.lit("XGB"),
                     F.lit("NN")),
            (F.col("value") % 3 + 1)
             .cast("int")
        ))
    .withColumn("score",
        (70 + F.col("value") % 30)
         .cast("float"))
)

# Stream-static join
enriched = rate_stream.join(
    model_metadata, "model_id", "left"
)

# 2. Stateful operations with mapGroupsWithState
print("\n=== STATEFUL CUSTOM AGGREGATIONS ===")
print("""
Beyond windowed aggregations:
mapGroupsWithState — maintain arbitrary state

Use case: session tracking, running metrics,
          complex event detection

from pyspark.sql.streaming import GroupState

def update_state(key, inputs, state):
    # key: group key (model_id)
    # inputs: new events in this batch
    # state: current state for this key

    current = state.getOption or initial_state
    updated = process(current, inputs)
    state.update(updated)

    if should_emit(updated):
        yield result_row

df.groupBy("model_id") \\
  .mapGroupsWithState(update_state)
""")

# 3. Multiple streaming queries
print("\n=== MULTIPLE CONCURRENT QUERIES ===")
print("""
Run multiple streaming queries simultaneously:

# Query 1: high-priority alerts
alert_query = (stream
    .filter(score < 60)
    .writeStream
    .format("delta")
    .option("checkpointLocation", "/tmp/cp1")
    .start("/tmp/alerts")
)

# Query 2: regular metrics
metrics_query = (stream
    .groupBy(window("ts", "1 min"))
    .agg(avg("score"))
    .writeStream
    .format("delta")
    .option("checkpointLocation", "/tmp/cp2")
    .start("/tmp/metrics")
)

# Monitor all queries
for q in spark.streams.active:
    print(q.name, q.status["message"])

# Wait for all to finish
spark.streams.awaitAnyTermination()
""")

# 4. Streaming deduplication
print("\n=== STREAMING DEDUPLICATION ===")
print("""
Problem: Kafka delivers at-least-once
→ Duplicate events possible

Solution: dropDuplicates() on streaming!

stream \\
  .withWatermark("event_time", "1 hour") \\
  .dropDuplicates(["event_id", "event_time"])

Rules:
→ Must have watermark for streaming dedup
→ Dedup window = watermark duration
→ Events outside watermark: state dropped
→ After watermark: no more dedup for old events

Use cases:
→ Kafka at-least-once deduplication
→ Retry-safe event processing
→ Idempotent streaming pipelines
""")

# 5. Delta Lake as streaming source
print("\n=== DELTA AS STREAMING SOURCE ===")
print("""
Delta Lake can be BOTH source AND sink!

# Read Delta changes as stream
stream = (spark.readStream
    .format("delta")
    .option("startingVersion", "latest")
    .load("/tmp/ml_experiments")
)

# Process only new/changed rows
stream \\
    .filter(score > 90) \\
    .writeStream \\
    .format("delta") \\
    .start("/tmp/high_performers")

Benefits:
→ ACID guarantees on streaming reads
→ Exactly-once with Delta ACID
→ Time travel on streaming data
→ Schema evolution handled automatically
→ No Kafka needed for internal pipelines!

This is the Databricks-native streaming pattern:
Delta → Spark Streaming → Delta
        (no Kafka required!)
""")

# Run a quick stream demo
query = (rate_stream
    .withWatermark("timestamp", "1 minute")
    .dropDuplicates(["model_id", "timestamp"])
    .writeStream
    .format("memory")
    .queryName("dedup_demo")
    .outputMode("append")
    .start()
)

import time
time.sleep(5)
result = spark.sql(
    "SELECT model_id, COUNT(*) as count "
    "FROM dedup_demo GROUP BY model_id"
)
result.show()
query.stop()

mlflow.set_experiment("phase2_streaming_adv")
with mlflow.start_run(
        run_name="advanced_streaming"):
    mlflow.log_param("patterns", [
        "stream_static_join",
        "stateful_aggregations",
        "concurrent_queries",
        "deduplication",
        "delta_as_source"
    ])
    print("\nAdvanced streaming logged!")
