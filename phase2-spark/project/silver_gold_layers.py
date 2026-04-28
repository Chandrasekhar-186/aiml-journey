# Phase 2 Project — Silver + Gold Layers
# Date: April 24, 2026
# Lakehouse: Bronze → Silver → Gold!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import mlflow

spark = SparkSession.builder \
    .appName("ModelMonitor_SilverGold") \
    .getOrCreate()

print("="*60)
print("Building Silver + Gold Layers")
print("="*60)

# ── SILVER LAYER ────────────────────────────────
print("\n=== SILVER LAYER ===")
print("Bronze → clean → validate → enrich")

# Read Bronze (batch for now)
try:
    bronze_df = spark.read.format("delta") \
        .load("/tmp/monitor_bronze")
    print(f"Bronze rows loaded: "
          f"{bronze_df.count()}")
except Exception:
    # Generate synthetic data if Bronze empty
    print("Generating synthetic bronze data...")
    bronze_df = spark.createDataFrame([
        (f"pred_{i}", "RF",
         float(70 + i%30), i%2,
         float(5 + i%45),
         F.current_timestamp())
        for i in range(1000)
    ], ["prediction_id", "model_name",
        "score", "ground_truth",
        "latency_ms", "event_time"])

# 1. Data quality checks
print("\nData quality checks:")
total = bronze_df.count()
null_scores = bronze_df.filter(
    F.col("score").isNull()
).count()
invalid_scores = bronze_df.filter(
    (F.col("score") < 0) |
    (F.col("score") > 100)
).count()
print(f"  Total rows:      {total}")
print(f"  Null scores:     {null_scores}")
print(f"  Invalid scores:  {invalid_scores}")
print(f"  Quality rate:    "
      f"{(total-null_scores-invalid_scores)/total:.1%}")

# 2. Silver transformations
silver_df = (bronze_df
    # Drop nulls + invalid
    .filter(F.col("score").isNotNull())
    .filter(
        (F.col("score") >= 0) &
        (F.col("score") <= 100)
    )
    # Add derived columns
    .withColumn("is_correct",
        (F.col("score") > 80) ==
        F.col("ground_truth").cast("boolean")
    )
    .withColumn("quality_tier",
        F.when(F.col("score") >= 90, "premium")
         .when(F.col("score") >= 75, "standard")
         .otherwise("basic")
    )
    .withColumn("latency_bucket",
        F.when(F.col("latency_ms") < 10, "fast")
         .when(F.col("latency_ms") < 30, "medium")
         .otherwise("slow")
    )
    # Add processing metadata
    .withColumn("silver_processed_at",
        F.current_timestamp())
    .withColumn("silver_version",
        F.lit("1.0"))
)

# 3. Write Silver as Delta
(silver_df.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("model_name")
    .option("mergeSchema", "true")
    .save("/tmp/monitor_silver")
)
print(f"\nSilver written: {silver_df.count()} rows")
print("Partitioned by model_name!")

# Verify Silver
silver_read = spark.read.format("delta") \
    .load("/tmp/monitor_silver")
print("\nSilver schema:")
silver_read.printSchema()
silver_read.show(3)

# ── GOLD LAYER ──────────────────────────────────
print("\n=== GOLD LAYER ===")
print("Silver → aggregate → business metrics")

# 1. Model performance metrics (Gold table 1)
model_metrics = (silver_read
    .groupBy("model_name")
    .agg(
        F.count("*").alias("total_predictions"),
        F.round(F.avg("score"), 4)
         .alias("avg_score"),
        F.round(F.stddev("score"), 4)
         .alias("score_stddev"),
        F.round(F.avg("latency_ms"), 2)
         .alias("avg_latency_ms"),
        F.round(
            F.sum(F.col("is_correct").cast("int")) /
            F.count("*"), 4
        ).alias("accuracy"),
        F.round(
            F.sum(
                (F.col("latency_bucket") == "slow")
                 .cast("int")
            ) / F.count("*"), 4
        ).alias("slow_rate")
    )
    .withColumn("gold_updated_at",
        F.current_timestamp())
)

(model_metrics.write
    .format("delta")
    .mode("overwrite")
    .save("/tmp/monitor_gold_metrics")
)
print("\nGold metrics:")
model_metrics.show()

# 2. Drift detection (Gold table 2)
print("\n=== DRIFT DETECTION ===")

# Calculate rolling statistics
window_spec = Window \
    .partitionBy("model_name") \
    .orderBy("silver_processed_at") \
    .rowsBetween(-99, 0)  # last 100 preds

drift_df = (silver_read
    .withColumn("rolling_avg_score",
        F.avg("score").over(window_spec))
    .withColumn("rolling_stddev",
        F.stddev("score").over(window_spec))
    .withColumn("score_zscore",
        (F.col("score") -
         F.col("rolling_avg_score")) /
        (F.col("rolling_stddev") + 0.001)
    )
    .withColumn("is_anomaly",
        F.abs(F.col("score_zscore")) > 2.5
    )
)

anomaly_rate = (drift_df
    .groupBy("model_name")
    .agg(
        F.round(
            F.avg(F.col("is_anomaly")
                   .cast("double")), 4
        ).alias("anomaly_rate"),
        F.count(
            F.when(F.col("is_anomaly"), 1)
        ).alias("anomaly_count")
    )
)

(anomaly_rate.write
    .format("delta")
    .mode("overwrite")
    .save("/tmp/monitor_gold_drift")
)
print("Drift detection Gold table:")
anomaly_rate.show()

# 3. Alert logic
alert_threshold = 0.05  # 5% anomaly rate
alerts = anomaly_rate.filter(
    F.col("anomaly_rate") > alert_threshold
)
alert_count = alerts.count()

print(f"\n🚨 Models with drift alerts: "
      f"{alert_count}")
if alert_count > 0:
    alerts.show()
else:
    print("✅ All models within normal range!")

# 4. Log everything to MLflow
mlflow.set_experiment("model_monitor_project")
with mlflow.start_run(
        run_name="silver_gold_layers"):
    mlflow.log_param("silver_partition",
                     "model_name")
    mlflow.log_param("drift_window", 100)
    mlflow.log_param("anomaly_threshold", 2.5)
    mlflow.log_metric("silver_rows",
                       silver_read.count())
    mlflow.log_metric("gold_models",
                       model_metrics.count())
    mlflow.log_metric("alert_count",
                       alert_count)
    print("\nSilver + Gold logged to MLflow!")

print("\n" + "="*60)
print("Lakehouse Architecture COMPLETE!")
print("Bronze ✅ → Silver ✅ → Gold ✅")
print("="*60)
