# Phase 3 CV Day 6 — CV at Scale with Spark
# Date: May 27, 2026
# Petabyte-scale computer vision!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow

spark = SparkSession.builder \
    .appName("CVAtScale") \
    .getOrCreate()

print("="*60)
print("Large-Scale CV on Apache Spark")
print("="*60)

"""
CV AT DATABRICKS SCALE

Problem: process 100M+ images
Standard PyTorch: sequential, one machine
Spark solution: distribute across 1000s of workers

Architecture:
Images in Delta Lake (binary/path)
    ↓ Spark reads + distributes
Each executor: runs model inference
    ↓ Results
Delta Lake (predictions/embeddings)
    ↓ Downstream analytics/search

Key tools:
1. Pandas UDF (vectorized): batch inference
2. Petastorm: PyTorch↔Spark data bridge
3. Horovod: distributed training on Spark
4. MLflow: track all experiments
"""

# 1. Pandas UDF for model inference
print("\n=== PANDAS UDF FOR CV INFERENCE ===")
print("""
# Standard approach: Pandas UDF
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import io

# Load model ONCE per executor (not per row!)
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = torch.hub.load(
            'pytorch/vision',
            'resnet50',
            pretrained=True
        )
        MODEL.eval()
    return MODEL

@F.pandas_udf(ArrayType(FloatType()))
def extract_features(image_bytes: pd.Series
                     ) -> pd.Series:
    model = get_model()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    features = []
    for img_bytes in image_bytes:
        img = Image.open(io.BytesIO(img_bytes))
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
        features.append(feat.numpy().tolist()[0])
    return pd.Series(features)

# Apply to Delta table!
df = spark.read.format("delta").load("/images")
df_features = df.withColumn(
    "features",
    extract_features(F.col("image_bytes"))
)
df_features.write.format("delta").save("/features")
""")

# 2. Distributed training with Spark
print("\n=== HOROVOD DISTRIBUTED TRAINING ===")
print("""
# Train PyTorch model on Spark cluster!
# Each worker trains on subset of data

import horovod.spark.torch as hvd
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import DBFSLocalStore

# Define model training function
def train_fn(model, optimizer, train_loader):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    return model

# Distributed training across Spark workers!
backend = SparkBackend(
    num_proc=8,  # 8 Spark executors
    stdout=sys.stdout,
    stderr=sys.stderr
)
store = DBFSLocalStore('/dbfs/tmp/horovod')

# Each worker trains on 1/8 of data!
torch_estimator = hvd.TorchEstimator(
    backend=backend,
    store=store,
    model=my_model,
    optimizer=my_optimizer,
    loss=nn.CrossEntropyLoss(),
    epochs=10,
    batch_size=64,
    feature_cols=['features'],
    label_cols=['label']
)

fitted_model = torch_estimator.fit(spark_df)
""")

# 3. Mosaic AI Vector Search
print("\n=== MOSAIC AI VECTOR SEARCH ===")
print("""
# Store CLIP embeddings in Databricks!
# Enables semantic image search

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# Create vector search index on Delta table
client.create_delta_sync_index(
    endpoint_name="my_endpoint",
    source_table_name="catalog.schema.embeddings",
    index_name="catalog.schema.image_index",
    pipeline_type="TRIGGERED",
    primary_key="image_id",
    embedding_dimension=512,
    embedding_vector_column="clip_embedding"
)

# Query: find similar images to text!
results = client.get_index(
    endpoint_name="my_endpoint",
    index_name="catalog.schema.image_index"
).similarity_search(
    columns=["image_id", "image_path"],
    query_vector=clip_text_embedding,
    num_results=10
)

# Returns top-10 most similar images!
# This powers Databricks AI search! 🚀
""")

# 4. Simulate CV pipeline in Spark
print("\n=== SIMULATED CV PIPELINE ===")

# Simulate image metadata table
image_data = [
    (f"img_{i:04d}",
     f"/images/img_{i:04d}.jpg",
     float(i % 100) / 100,  # fake confidence
     i % 10)                 # fake class
    for i in range(1000)
]
schema = StructType([
    StructField("image_id", StringType()),
    StructField("path", StringType()),
    StructField("confidence", FloatType()),
    StructField("predicted_class", IntegerType())
])
df = spark.createDataFrame(image_data, schema)

# Analytics on predictions
print("Class distribution:")
df.groupBy("predicted_class") \
  .agg(
      F.count("*").alias("count"),
      F.round(F.avg("confidence"), 3)
       .alias("avg_conf")
  ).orderBy("predicted_class").show()

# Low confidence predictions → review queue
low_conf = df.filter(F.col("confidence") < 0.5)
print(f"Low confidence images: "
      f"{low_conf.count()}")

# Log to MLflow
mlflow.set_experiment("phase3_cv_spark")
with mlflow.start_run(
        run_name="CV_at_scale"):
    mlflow.log_params({
        "framework": "PySpark",
        "inference": "Pandas_UDF",
        "storage": "Delta_Lake",
        "embeddings": "CLIP_512d",
        "search": "Vector_Search"
    })
    mlflow.log_metric(
        "total_images", df.count()
    )
    mlflow.log_metric(
        "low_conf_count", low_conf.count()
    )
    print("\nCV at scale logged to MLflow!")

print("\n" + "="*60)
print("Large-scale CV on Spark — MASTERED!")
print("="*60)
