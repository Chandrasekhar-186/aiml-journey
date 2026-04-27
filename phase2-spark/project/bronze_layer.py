# Phase 2 Project — Bronze Layer
# Date: April 23, 2026

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow

spark = SparkSession.builder \
    .appName("ModelMonitor_Bronze") \
    .getOrCreate()

print("Building Bronze Layer...")

# Simulate streaming ML predictions
raw_stream = (spark.readStream
    .format("rate")
    .option("rowsPerSecond", 20)
    .load()
    .withColumn("prediction_id",
        F.expr("uuid()"))
    .withColumn("model_name",
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
    .withColumn("ground_truth",
        ((F.col("value") % 2) == 0)
         .cast("int"))
    .withColumn("latency_ms",
        (5 + F.col("value") % 45)
         .cast("float"))
    .withColumn("event_time",
        F.col("timestamp"))
    .drop("value")
)

# Write to Bronze Delta Lake
query = (raw_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation",
            "/tmp/monitor_bronze_checkpoint")
    .trigger(processingTime="10 seconds")
    .start("/tmp/monitor_bronze")
)

import time
print("Bronze layer streaming for 30 seconds...")
time.sleep(30)
query.stop()

# Verify
bronze_df = spark.read.format("delta") \
    .load("/tmp/monitor_bronze")
print(f"Bronze rows: {bronze_df.count()}")
bronze_df.show(5)

mlflow.set_experiment("model_monitor_project")
with mlflow.start_run(run_name="bronze_layer"):
    mlflow.log_param("layer", "bronze")
    mlflow.log_metric("rows_ingested",
                       bronze_df.count())
    print("Bronze layer logged to MLflow!")
