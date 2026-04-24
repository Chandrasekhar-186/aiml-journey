# Phase 2 Day 9 — Kafka + Stateful Streaming
# Date: April 19, 2026
# Build real-time ML prediction pipeline!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow

spark = SparkSession.builder \
    .appName("KafkaStatefulStreaming") \
    .config("spark.sql.adaptive.enabled",
            "true") \
    .getOrCreate()

print("="*60)
print("Kafka + Stateful Streaming Pipeline")
print("="*60)

"""
REAL-TIME ML PIPELINE ARCHITECTURE:

Mobile/Web Events
      ↓
Kafka Topic: "ml-events"
      ↓
Spark Structured Streaming
      ↓ (parse + validate)
Bronze Delta Table
      ↓ (feature engineering)
Silver Delta Table
      ↓ (MLflow model scoring)
Gold Delta Table (predictions)
      ↓
Kafka Topic: "ml-predictions"
      ↓
Downstream consumers
"""

# 1. Simulate Kafka source (rate source)
print("\n=== SIMULATING KAFKA SOURCE ===")
raw_stream = (spark.readStream
    .format("rate")
    .option("rowsPerSecond", 50)
    .load()
    .withColumn("key",
        (F.col("value") % 100)
         .cast("string"))
    .withColumn("value",
        F.to_json(F.struct(
            (F.col("value") % 5)
             .alias("model_id"),
            (70 + F.col("value") % 30)
             .cast("float").alias("score"),
            F.col("timestamp").alias("event_time"),
            (F.col("value") % 3)
             .alias("user_segment")
        )))
)

print(f"Is streaming: {raw_stream.isStreaming}")

# 2. Parse JSON schema
event_schema = StructType([
    StructField("model_id", IntegerType()),
    StructField("score", FloatType()),
    StructField("event_time", TimestampType()),
    StructField("user_segment", IntegerType())
])

parsed_stream = (raw_stream
    .select(
        F.col("key"),
        F.from_json(
            F.col("value"), event_schema
        ).alias("data"),
        F.col("timestamp").alias("kafka_time")
    )
    .select(
        "key",
        "data.model_id",
        "data.score",
        "data.event_time",
        "data.user_segment",
        "kafka_time"
    )
    .filter(F.col("score").isNotNull())
    .filter(F.col("score") > 0)
)

# 3. Stateless transformations
enriched = (parsed_stream
    .withColumn("quality_tier",
        F.when(F.col("score") >= 90, "premium")
         .when(F.col("score") >= 75, "standard")
         .otherwise("basic")
    )
    .withColumn("processing_time",
        F.current_timestamp())
    .withColumn("latency_ms",
        (F.unix_timestamp("processing_time") -
         F.unix_timestamp("kafka_time")) * 1000
    )
)

# 4. STATEFUL — Windowed aggregations
print("\n=== STATEFUL STREAMING ===")
print("""
Stateful operations maintain state
across micro-batches:

Windowed aggregations:
→ Count events per model per minute
→ Calculate running averages
→ Detect anomalies over time

State is stored in:
→ Executor memory (fast)
→ Checkpoint location (fault tolerant)

Without watermark:
→ State grows forever → OOM!

With watermark:
→ Old state cleaned up automatically
→ Bounded memory usage ✅
""")

# Tumbling window — non-overlapping
tumbling = (enriched
    .withWatermark("event_time", "2 minutes")
    .groupBy(
        F.window("event_time", "1 minute"),
        "model_id"
    )
    .agg(
        F.count("*").alias("event_count"),
        F.avg("score").alias("avg_score"),
        F.min("score").alias("min_score"),
        F.max("score").alias("max_score")
    )
)

# Sliding window — overlapping
sliding = (enriched
    .withWatermark("event_time", "5 minutes")
    .groupBy(
        F.window("event_time",
                  "2 minutes",    # window size
                  "30 seconds"),  # slide interval
        "quality_tier"
    )
    .agg(
        F.count("*").alias("count"),
        F.avg("score").alias("avg_score")
    )
)

# Session window (Spark 3.2+)
print("""
Session Window (dynamic duration):
Groups events that are close in time
Gap = period of inactivity

.groupBy(
    session_window("event_time", "5 minutes"),
    "user_id"
)
→ Events within 5-min gaps = same session
→ Perfect for user behavior analysis!
""")

# 5. foreachBatch — most powerful sink!
print("\n=== foreachBatch SINK ===")

def process_batch(batch_df, batch_id):
    """
    Process each micro-batch with full
    batch DataFrame operations!
    """
    if batch_df.isEmpty():
        return

    # 1. Write to Delta Lake (Bronze)
    batch_df.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save("/tmp/bronze_stream")

    # 2. Score with MLflow model
    # model = mlflow.sklearn.load_model(
    #     "models:/MLModel/Production"
    # )
    # predictions = model.predict(features)

    # 3. Log batch metrics to MLflow
    count = batch_df.count()
    avg_score = batch_df.agg(
        F.avg("score")
    ).collect()[0][0]

    with mlflow.start_run(
            run_name=f"batch_{batch_id}",
            nested=True):
        mlflow.log_metric("batch_count", count)
        mlflow.log_metric("avg_score",
                           avg_score or 0)
        mlflow.log_metric("batch_id", batch_id)

    print(f"Batch {batch_id}: "
          f"{count} rows, "
          f"avg_score={avg_score:.2f}")

# Start streaming query
query = (enriched
    .writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation",
            "/tmp/stream_checkpoint")
    .trigger(processingTime="5 seconds")
    .start()
)

import time
print("\nStreaming for 15 seconds...")
time.sleep(15)
query.stop()
print("Streaming complete!")

# 6. Query Bronze table
print("\n=== BRONZE TABLE RESULTS ===")
try:
    bronze = spark.read.format("delta") \
        .load("/tmp/bronze_stream")
    print(f"Bronze rows: {bronze.count()}")
    bronze.show(5)
except Exception:
    print("Bronze table not yet created")

# 7. Streaming best practices
print("\n=== STREAMING BEST PRACTICES ===")
print("""
1. ALWAYS set checkpointLocation
   → Enables fault tolerance
   → Exactly-once processing

2. ALWAYS add watermark for aggregations
   → Bounds state size
   → Prevents OOM

3. Use foreachBatch for complex logic
   → Full DataFrame API available
   → Can write to multiple sinks
   → Can call external APIs

4. Tune trigger interval
   → Lower: lower latency, higher overhead
   → Higher: higher latency, more efficient
   → Start with 30s, tune based on SLA

5. Monitor streaming queries
   spark.streams.active  ← all active queries
   query.lastProgress    ← latest batch stats
   query.status          ← current status

6. Use Delta Lake as streaming sink
   → ACID guarantees for streaming writes
   → Enables downstream batch queries
   → Time travel on streaming data!
""")

mlflow.set_experiment("phase2_streaming")
with mlflow.start_run(
        run_name="kafka_stateful_pipeline"):
    mlflow.log_param("trigger", "5 seconds")
    mlflow.log_param("watermark", "2 minutes")
    mlflow.log_param("sink", "delta_lake")
    print("\nStreaming pipeline logged!")
