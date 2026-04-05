# Day 21 — Spark Certification Intensive
# Date: April 2, 2026
# Target: Pass Spark Associate cert in Phase 2!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

# ── SECTION 1: DataFrame API (30% of exam) ──────

spark = SparkSession.builder \
    .appName("SparkCertPrep") \
    .getOrCreate()

# Q: Create DataFrame with explicit schema
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("score", DoubleType(), True),
    StructField("dept", StringType(), True)
])
data = [(1,"Alice",92.5,"ML"),
        (2,"Bob",88.0,"DE"),
        (3,"Carol",95.5,"ML"),
        (4,"Dave",78.0,"DE"),
        (5,"Eve",91.0,"ML")]
df = spark.createDataFrame(data, schema)

# Q: Select + filter + alias
result = (df
    .select("name",
            F.col("score").alias("accuracy"),
            "dept")
    .filter(F.col("score") > 85)
    .orderBy(F.desc("score"))
)
result.show()

# Q: GroupBy + multiple aggregations
df.groupBy("dept").agg(
    F.count("*").alias("count"),
    F.round(F.avg("score"), 2).alias("avg"),
    F.max("score").alias("max"),
    F.min("score").alias("min"),
    F.stddev("score").alias("std")
).show()

# ── SECTION 2: Spark SQL (25% of exam) ──────────

df.createOrReplaceTempView("employees")

spark.sql("""
    SELECT dept,
           COUNT(*) as headcount,
           ROUND(AVG(score), 2) as avg_score,
           RANK() OVER (
               ORDER BY AVG(score) DESC
           ) as dept_rank
    FROM employees
    GROUP BY dept
""").show()

# ── SECTION 3: Optimization (20% of exam) ───────

# Q: Explain the difference
print("TRANSFORMATION vs ACTION:")
print("Transformations (lazy):")
print("  filter(), select(), groupBy(),")
print("  join(), withColumn(), orderBy()")
print("\nActions (trigger execution):")
print("  show(), count(), collect(),")
print("  write(), first(), take()")

# Q: When does a shuffle happen?
print("\nOperations that cause SHUFFLE:")
print("  groupBy(), join(), distinct(),")
print("  repartition(), orderBy()")

# Q: Partition optimization
df_repartitioned = df.repartition(4, "dept")
print(f"\nPartitions: "
      f"{df_repartitioned.rdd.getNumPartitions()}")

# Coalesce — reduce partitions (no shuffle!)
df_coalesced = df_repartitioned.coalesce(2)
print(f"After coalesce: "
      f"{df_coalesced.rdd.getNumPartitions()}")

# ── SECTION 4: Delta Lake (15% of exam) ─────────

# Q: Delta Lake operations
print("\nDelta Lake key concepts:")
print("""
ACID:         Atomicity, Consistency,
              Isolation, Durability
Time travel:  VERSION AS OF n
              TIMESTAMP AS OF '2026-01-01'
OPTIMIZE:     Compacts small files
ZORDER:       Co-locates related data
VACUUM:       Removes old versions
MERGE:        Upsert operation
""")

# ── SECTION 5: Spark Streaming (10% of exam) ────

print("Streaming key concepts:")
print("""
readStream:   creates streaming DataFrame
writeStream:  outputs stream to sink
trigger:      processing interval
outputMode:
  append:   only new rows (default)
  complete: all rows (aggregations)
  update:   only changed rows
checkpointLocation: fault tolerance!
""")

# ── 30 PRACTICE QUESTIONS ───────────────────────
questions = """
SPARK CERT PRACTICE — 30 QUESTIONS

DataFrame API:
Q1:  What does repartition() vs coalesce() do?
Q2:  How do you add a new column in Spark?
Q3:  What is the difference between map() and flatMap()?
Q4:  How do you handle null values in Spark?
Q5:  What does cache() vs persist() do?

Spark SQL:
Q6:  How do you register a DataFrame as a temp view?
Q7:  What is the difference between RANK and DENSE_RANK?
Q8:  How do you use window functions in Spark SQL?
Q9:  What does EXPLAIN do?
Q10: How do you read a CSV with a header in Spark?

Optimization:
Q11: What is the Catalyst optimizer?
Q12: When should you use broadcast join?
Q13: What causes a shuffle in Spark?
Q14: What is data skew and how do you fix it?
Q15: What is AQE (Adaptive Query Execution)?

Delta Lake:
Q16: What makes Delta Lake ACID compliant?
Q17: How do you query a previous version?
Q18: What is OPTIMIZE ZORDER?
Q19: What does VACUUM do?
Q20: How does MERGE work in Delta Lake?

Streaming:
Q21: What is checkpointing used for?
Q22: What are the 3 output modes?
Q23: What is a watermark in Spark Streaming?
Q24: How do you read from Kafka in Spark?
Q25: What is micro-batch processing?

Architecture:
Q26: What is a DAG in Spark?
Q27: What is the difference between a job, stage, task?
Q28: What is lazy evaluation?
Q29: What is the driver vs executor?
Q30: What is a partition in Spark?
"""
print(questions)

# Save your answers!
print("Write your answers in spark_cert_answers.md")
print("Score yourself — target 27+/30!")
