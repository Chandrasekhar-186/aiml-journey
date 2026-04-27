# Phase 2 Day 12 — GraphFrames + Production Spark
# Date: April 22, 2026
# GraphFrames = graph algorithms at Spark scale!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import mlflow

spark = SparkSession.builder \
    .appName("GraphFramesProduction") \
    .config("spark.jars.packages",
            "graphframes:graphframes:0.8.2"
            "-spark3.0-s_2.12") \
    .getOrCreate()

print("="*60)
print("GraphFrames + Production Spark Patterns")
print("="*60)

# 1. GraphFrames introduction
print("\n=== GRAPHFRAMES OVERVIEW ===")
print("""
GraphFrames = Apache Spark + Graph algorithms

Built on DataFrames (not RDDs like GraphX)
→ Full Spark optimizer support
→ Python, Scala, Java APIs
→ Integrates with MLflow naturally

Key algorithms:
→ PageRank:          rank nodes by importance
→ BFS:               shortest path between nodes
→ Connected Components: find isolated clusters
→ Triangle Count:    measure network density
→ Label Propagation: community detection
→ Shortest Paths:    distance to landmark nodes

Use cases at Databricks scale:
→ Fraud detection (suspicious transaction networks)
→ Recommendation systems (user-item graphs)
→ Social network analysis (influencer detection)
→ Knowledge graphs (entity relationships)
→ Supply chain optimization
""")

try:
    from graphframes import GraphFrame

    # 2. Create a graph
    print("\n=== CREATING A GRAPH ===")

    # Vertices (nodes) — ML models as nodes
    vertices = spark.createDataFrame([
        ("RF", "RandomForest", 88.0),
        ("XGB", "XGBoost", 92.0),
        ("NN", "NeuralNet", 95.0),
        ("LR", "LogisticReg", 78.0),
        ("SVM", "SVM", 82.0),
        ("CNN", "CNN", 91.0),
    ], ["id", "name", "accuracy"])

    # Edges — model relationships
    edges = spark.createDataFrame([
        ("RF", "XGB", "ensemble_with"),
        ("XGB", "NN", "stacks_into"),
        ("NN", "CNN", "variant_of"),
        ("LR", "SVM", "similar_to"),
        ("RF", "LR", "baseline_for"),
        ("XGB", "CNN", "compared_to"),
        ("NN", "RF", "outperforms"),
    ], ["src", "dst", "relationship"])

    g = GraphFrame(vertices, edges)

    print(f"Vertices: {g.vertices.count()}")
    print(f"Edges:    {g.edges.count()}")
    g.vertices.show()
    g.edges.show()

    # 3. PageRank — find most important models
    print("\n=== PAGERANK ===")
    results = g.pageRank(
        resetProbability=0.15,
        maxIter=10
    )
    results.vertices \
        .select("id", "name",
                "accuracy", "pagerank") \
        .orderBy(F.desc("pagerank")) \
        .show()
    print("Most referenced/influential models ↑")

    # 4. BFS — find path between models
    print("\n=== BFS SHORTEST PATH ===")
    paths = g.bfs(
        fromExpr="id = 'RF'",
        toExpr="id = 'CNN'",
        maxPathLength=5
    )
    paths.show(truncate=False)

    # 5. Connected Components
    print("\n=== CONNECTED COMPONENTS ===")
    spark.sparkContext.setCheckpointDir(
        "/tmp/graphframes_checkpoint"
    )
    cc = g.connectedComponents()
    cc.select("id", "component").show()
    print(f"Unique components: "
          f"{cc.select('component').distinct().count()}")

    # 6. Triangle Count
    print("\n=== TRIANGLE COUNT ===")
    tc = g.triangleCount()
    tc.select("id", "count").show()

    # 7. Label Propagation (community detection)
    print("\n=== COMMUNITY DETECTION ===")
    lp = g.labelPropagation(maxIter=5)
    lp.select("id", "label").show()

except ImportError:
    print("GraphFrames not installed locally")
    print("Available in Databricks CE natively!")
    print("""
    # In Databricks notebook:
    # %pip install graphframes

    # Or in cluster config:
    # com.graphframes:graphframes:0.8.2-spark3.0-s_2.12

    # Then use exactly as shown above!
    """)

# 8. Production Spark patterns
print("\n=== PRODUCTION SPARK PATTERNS ===")

# Pattern 1: Schema enforcement
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
    IntegerType, TimestampType
)

schema = StructType([
    StructField("model_id",
                StringType(), False),  # not null!
    StructField("accuracy",
                DoubleType(), True),
    StructField("created_at",
                TimestampType(), True)
])

print("Pattern 1: Always define schema!")
print("→ Prevents schema inference on large data")
print("→ Catches bad data at read time")
print("→ Required for streaming DataFrames")

# Pattern 2: Error handling in Spark
print("\nPattern 2: Handle bad records!")
print("""
spark.read.csv(path)
  .option("mode", "PERMISSIVE")   # keep bad rows
  .option("columnNameOfCorruptRecord", "_corrupt")
  # OR
  .option("mode", "DROPMALFORMED") # drop bad rows
  # OR
  .option("mode", "FAILFAST")     # fail on bad row
""")

# Pattern 3: Dynamic partition overwrite
print("Pattern 3: Dynamic partition overwrite!")
print("""
spark.conf.set(
    "spark.sql.sources.partitionOverwriteMode",
    "dynamic"
)
# Now writing to partition only overwrites THAT partition
# Not the entire table!
df.write.mode("overwrite")
       .partitionBy("date")
       .save(path)
→ Only overwrites date=2026-04-22 partition
→ Other dates untouched ✅
""")

# Pattern 4: Idempotent writes with Delta
print("Pattern 4: Idempotent writes!")
print("""
# Add unique transaction ID
df.withColumn("txn_id", F.expr("uuid()"))

# Write with MERGE — safe for retries!
DeltaTable.forPath(spark, path) \\
    .alias("old") \\
    .merge(df.alias("new"),
           "old.txn_id = new.txn_id") \\
    .whenNotMatchedInsertAll() \\
    .execute()
→ Running twice = same result ✅
→ No duplicate rows!
""")

# Pattern 5: Resource management
print("Pattern 5: Resource management!")
print("""
# Set job-specific config
spark.conf.set(
    "spark.executor.cores", "4"
)

# Use context manager for temp config
with spark._jvm.org.apache.spark.util \
        .Utils.withSpark(spark):
    # config restored after block
    pass

# Monitor active queries
for query in spark.streams.active:
    print(query.name, query.status)
""")

# Log to MLflow
mlflow.set_experiment("phase2_production")
with mlflow.start_run(
        run_name="graphframes_production"):
    mlflow.log_param("topic",
                     "graphframes_production")
    mlflow.log_metric("patterns_covered", 5)
    print("\nGraphFrames + production logged!")
