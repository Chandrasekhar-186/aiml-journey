# CV Project — Production Detection Pipeline
# Date: May 28, 2026
# YOLOv8 + Spark + Delta Lake + MLflow

import torch
import numpy as np
from ultralytics import YOLO
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow
import mlflow.pytorch
import pandas as pd
import time
import json

print("="*60)
print("Production CV Pipeline")
print("YOLOv8 + Spark + Delta + MLflow")
print("="*60)

# ── SETUP ────────────────────────────────────
spark = SparkSession.builder \
    .appName("CVPipeline") \
    .config("spark.sql.adaptive.enabled",
            "true") \
    .getOrCreate()

device = 'cuda' if torch.cuda.is_available() \
          else 'cpu'
print(f"Device: {device}")

# ── BRONZE LAYER — Raw image metadata ────────
print("\n=== BRONZE LAYER ===")

# Simulate image dataset
# In production: read actual image paths from S3
image_data = []
import random
random.seed(42)

sample_images = [
    "bus.jpg", "dog.jpg", "cat.jpg",
    "person.jpg", "car.jpg"
]
for i in range(100):
    image_data.append({
        "image_id": f"img_{i:04d}",
        "image_path": f"s3://bucket/images/"
                      f"{sample_images[i%5]}",
        "source": "webcam" if i < 50
                  else "upload",
        "ingested_at": "2026-05-28",
        "batch_id": i // 10
    })

bronze_schema = StructType([
    StructField("image_id", StringType()),
    StructField("image_path", StringType()),
    StructField("source", StringType()),
    StructField("ingested_at", StringType()),
    StructField("batch_id", IntegerType())
])

bronze_df = spark.createDataFrame(
    image_data, bronze_schema
)
bronze_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/cv_bronze")

print(f"Bronze: {bronze_df.count()} images")
bronze_df.show(3)

# ── YOLOV8 INFERENCE ─────────────────────────
print("\n=== YOLOV8 DISTRIBUTED INFERENCE ===")

# Detection result schema
detection_schema = ArrayType(StructType([
    StructField("class_id", IntegerType()),
    StructField("class_name", StringType()),
    StructField("confidence", FloatType()),
    StructField("x1", FloatType()),
    StructField("y1", FloatType()),
    StructField("x2", FloatType()),
    StructField("y2", FloatType()),
]))

# Global model (loaded once per executor!)
_model = None

def get_yolo_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8n.pt')
    return _model

@F.pandas_udf(detection_schema)
def run_yolov8_inference(
    image_paths: pd.Series
) -> pd.Series:
    """
    Distributed YOLOv8 inference.
    Each executor runs this on its partition.
    Model loaded ONCE per executor!
    """
    model = get_yolo_model()
    all_detections = []

    for path in image_paths:
        try:
            # In production: load actual image
            # Here: use sample URL
            results = model.predict(
                source='https://ultralytics.com'
                        '/images/bus.jpg',
                conf=0.25,
                iou=0.7,
                verbose=False
            )

            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class_id": int(
                            box.cls[0]
                        ),
                        "class_name": model.names[
                            int(box.cls[0])
                        ],
                        "confidence": float(
                            box.conf[0]
                        ),
                        "x1": float(
                            box.xyxy[0][0]
                        ),
                        "y1": float(
                            box.xyxy[0][1]
                        ),
                        "x2": float(
                            box.xyxy[0][2]
                        ),
                        "y2": float(
                            box.xyxy[0][3]
                        ),
                    })
            all_detections.append(detections)
        except Exception as e:
            all_detections.append([])

    return pd.Series(all_detections)

# Run inference
print("Running YOLOv8 on Bronze images...")
start_time = time.time()

inference_df = bronze_df.withColumn(
    "detections",
    run_yolov8_inference(F.col("image_path"))
)
inference_df.cache()
total_inferred = inference_df.count()
elapsed = time.time() - start_time

print(f"Inferred {total_inferred} images "
      f"in {elapsed:.1f}s")

# ── SILVER LAYER — Parsed detections ─────────
print("\n=== SILVER LAYER ===")

# Explode detections to one row per detection
silver_df = (inference_df
    .withColumn("detection",
        F.explode_outer("detections"))
    .select(
        "image_id", "image_path",
        "source", "ingested_at", "batch_id",
        F.col("detection.class_id"),
        F.col("detection.class_name"),
        F.col("detection.confidence"),
        F.col("detection.x1"),
        F.col("detection.y1"),
        F.col("detection.x2"),
        F.col("detection.y2"),
    )
    .filter(
        F.col("confidence").isNotNull()
    )
    # Add computed columns
    .withColumn("box_area",
        (F.col("x2") - F.col("x1")) *
        (F.col("y2") - F.col("y1"))
    )
    .withColumn("quality_tier",
        F.when(F.col("confidence") >= 0.8,
               "high")
         .when(F.col("confidence") >= 0.5,
               "medium")
         .otherwise("low")
    )
    .withColumn("processed_at",
        F.current_timestamp())
)

silver_df.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("class_name") \
    .save("/tmp/cv_silver")

silver_count = silver_df.count()
print(f"Silver: {silver_count} detections")
print("\nDetections per class:")
silver_df.groupBy("class_name") \
    .agg(
        F.count("*").alias("detections"),
        F.round(F.avg("confidence"),
                 3).alias("avg_conf")
    ).orderBy(F.desc("detections")).show()

# ── GOLD LAYER — Aggregated stats ────────────
print("\n=== GOLD LAYER ===")

gold_df = silver_df.groupBy(
    "class_name", "quality_tier"
).agg(
    F.count("*").alias("total_detections"),
    F.round(F.avg("confidence"), 4)
     .alias("avg_confidence"),
    F.round(F.avg("box_area"), 2)
     .alias("avg_box_area"),
    F.countDistinct("image_id")
     .alias("images_with_class"),
    F.round(F.stddev("confidence"), 4)
     .alias("conf_stddev")
).withColumn("gold_updated_at",
    F.current_timestamp()
)

gold_df.write.format("delta") \
    .mode("overwrite") \
    .save("/tmp/cv_gold")

print("Gold layer stats:")
gold_df.orderBy(
    F.desc("total_detections")
).show()

# ── MLFLOW TRACKING ───────────────────────────
print("\n=== MLFLOW EXPERIMENT TRACKING ===")

total_detections = silver_count
high_conf = silver_df.filter(
    F.col("confidence") >= 0.8
).count()
unique_classes = silver_df.select(
    "class_name"
).distinct().count()

mlflow.set_experiment("cv_project_pipeline")
with mlflow.start_run(
        run_name="YOLOv8_production"):
    # Log pipeline config
    mlflow.log_params({
        "model": "YOLOv8n",
        "conf_threshold": 0.25,
        "iou_threshold": 0.7,
        "architecture": "CSPDarknet+PAN-FPN",
        "inference": "Spark_PandasUDF",
        "storage": "Delta_Lakehouse"
    })

    # Log metrics
    mlflow.log_metrics({
        "total_images": total_inferred,
        "total_detections": total_detections,
        "high_conf_detections": high_conf,
        "unique_classes": unique_classes,
        "inference_time_sec": elapsed,
        "images_per_second":
            total_inferred / max(elapsed, 0.1),
        "avg_detections_per_image":
            total_detections /
            max(total_inferred, 1)
    })

    # Log model info
    mlflow.set_tag("model_type",
                    "object_detection")
    mlflow.set_tag("framework", "YOLOv8")
    mlflow.set_tag("pipeline",
                    "bronze_silver_gold")

    # Save Gold stats as artifact
    gold_pd = gold_df.toPandas()
    gold_pd.to_csv("/tmp/gold_stats.csv",
                    index=False)
    mlflow.log_artifact("/tmp/gold_stats.csv")

    print(f"✅ Pipeline metrics logged!")
    print(f"   Images processed: {total_inferred}")
    print(f"   Total detections: {total_detections}")
    print(f"   Unique classes:   {unique_classes}")
    print(f"   High confidence:  {high_conf}")

# ── SUMMARY ──────────────────────────────────
print("\n" + "="*60)
print("CV PIPELINE COMPLETE! 🎯")
print("="*60)
print("""
Architecture built:
✅ Bronze:  Raw image metadata in Delta
✅ Silver:  Per-detection rows + features
✅ Gold:    Class statistics + quality
✅ MLflow:  Full experiment tracking

Databricks competencies demonstrated:
1. YOLOv8 (production detection)
2. Apache Spark (distributed inference)
3. Delta Lake (Lakehouse architecture)
4. MLflow (tracking + artifacts)
5. Pandas UDF (bridge PyTorch ↔ Spark)
6. Production patterns (partitioning, cache)
7. CV metrics (confidence, box area, quality)
""")
