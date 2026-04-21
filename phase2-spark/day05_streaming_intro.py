# Phase 2 Day 5 — Spark Structured Streaming
# Date: April 15, 2026

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("StreamingIntro") \
    .getOrCreate()

print("="*55)
print("Spark Structured Streaming")
print("="*55)

"""
STREAMING MENTAL MODEL:

Source (infinite) → Transformations → Sink

Spark treats stream as UNBOUNDED TABLE:
→ New data = new rows appended
→ Same DataFrame API works!
→ Just add .readStream + .writeStream

Micro-batch processing:
Every trigger interval → process new data
→ Default: process ASAP
→ Can set: .trigger(processingTime="1 minute")
"""

# 1. Read from rate source (built-in generator)
streaming_df = (spark.readStream
    .format("rate")
    .option("rowsPerSecond", 100)
    .load()
)
print(f"Is streaming: {streaming_df.isStreaming}")
print(f"Schema: {streaming_df.schema}")

# 2. Apply transformations (same as batch!)
processed = (streaming_df
    .withColumn("model_id",
        F.col("value") % 5)
    .withColumn("score",
        (70 + F.col("value") % 30).cast("float"))
    .withColumn("is_good",
        F.col("score") > 85)
)

# 3. Output modes
print("\n=== OUTPUT MODES ===")
print("""
append:   Only new rows added since last trigger
          → Good for: event logging, inserts
          → Cannot use with aggregations!

complete: Entire result table every trigger
          → Good for: aggregations, counts
          → Can be expensive for large results

update:   Only rows that changed since trigger
          → Good for: aggregations with updates
          → Most efficient for stateful ops
""")

# 4. Trigger types
print("=== TRIGGER TYPES ===")
print("""
Default (unset):
  Process ASAP — continuous micro-batches

processingTime:
  .trigger(processingTime="30 seconds")
  Process every 30 seconds

once:
  .trigger(once=True)
  Process all available data, then stop
  Great for batch-style streaming jobs!

availableNow: (Spark 3.3+)
  .trigger(availableNow=True)
  Like once but can split into multiple batches
""")

# 5. Windowing operations
print("=== WINDOWED AGGREGATIONS ===")
windowed = (streaming_df
    .withColumn("score",
        (70 + F.col("value") % 30).cast("float"))
    .groupBy(
        F.window("timestamp", "1 minute",
                  "30 seconds"),  # sliding window
        (F.col("value") % 5).alias("model_id")
    )
    .agg(
        F.avg("score").alias("avg_score"),
        F.count("*").alias("count")
    )
)

# 6. Watermarks — handle late data!
print("\n=== WATERMARKS ===")
print("""
Problem: Late arriving data in streams
Example: Mobile app sends event 5 min late

Without watermark:
→ Spark keeps all windows forever
→ OOM eventually!

With watermark:
.withWatermark("timestamp", "10 minutes")
→ "I'll wait up to 10 min for late data"
→ Drop data older than watermark
→ Enables state cleanup = bounded memory!

Best practice: ALWAYS add watermark
               for streaming aggregations!
""")

# Write stream example
query = (processed
    .writeStream
    .format("memory")
    .queryName("stream_results")
    .outputMode("append")
    .start()
)

import time
time.sleep(3)
spark.sql("SELECT * FROM stream_results "
          "LIMIT 5").show()
query.stop()
print("Streaming query complete!")
