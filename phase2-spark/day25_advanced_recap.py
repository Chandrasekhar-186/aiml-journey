# Phase 2 Day 25 — Advanced Spark Recap
# Date: May 5, 2026
# Final consolidation before Phase 3!

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import mlflow

spark = SparkSession.builder \
    .appName("AdvancedRecap") \
    .getOrCreate()

print("="*60)
print("Phase 2 Final Recap — Advanced Topics")
print("="*60)

# 1. Spark SQL advanced functions
print("\n=== ADVANCED SQL FUNCTIONS ===")

data = [(1,"RF",88.0,"2026-04-01",
         [85,88,90,87]),
        (2,"XGB",92.0,"2026-04-02",
         [90,92,94,91]),
        (3,"NN",95.0,"2026-04-03",
         [93,95,96,94]),
        (4,"LR",78.0,"2026-04-04",
         [75,78,80,77])]

df = spark.createDataFrame(
    data, ["id","model","accuracy",
           "date","score_history"]
)

# Array functions
result = (df
    .withColumn("history_avg",
        F.aggregate(
            "score_history",
            F.lit(0.0),
            lambda acc, x: acc + x
        ) / F.size("score_history")
    )
    .withColumn("history_max",
        F.array_max("score_history"))
    .withColumn("history_min",
        F.array_min("score_history"))
    .withColumn("score_trend",
        F.element_at("score_history", -1) -
        F.element_at("score_history", 1)
    )
)
result.select(
    "model","history_avg",
    "history_max","score_trend"
).show()

# 2. Complex window operations
print("\n=== COMPLEX WINDOW OPERATIONS ===")
window_all = Window.orderBy("accuracy")
window_model = Window.partitionBy("model") \
    .orderBy("accuracy")
window_rolling = Window \
    .orderBy("date") \
    .rowsBetween(-2, 0)

df.withColumn("pct_rank",
    F.percent_rank().over(window_all)
).withColumn("ntile_4",
    F.ntile(4).over(window_all)
).withColumn("running_avg",
    F.avg("accuracy").over(window_rolling)
).select(
    "model","accuracy",
    "pct_rank","ntile_4","running_avg"
).show()

# 3. Pivot + unpivot
print("\n=== PIVOT OPERATIONS ===")
metrics = spark.createDataFrame([
    ("RF","accuracy",88.0),
    ("RF","f1_score",0.87),
    ("XGB","accuracy",92.0),
    ("XGB","f1_score",0.91),
    ("NN","accuracy",95.0),
    ("NN","f1_score",0.94),
], ["model","metric","value"])

# Pivot: rows → columns
pivoted = (metrics
    .groupBy("model")
    .pivot("metric")
    .agg(F.first("value"))
)
print("Pivoted:")
pivoted.show()

# Unpivot: columns → rows (stack trick)
unpivoted = pivoted.select(
    "model",
    F.expr("stack(2, 'accuracy', accuracy, "
           "'f1_score', f1_score) "
           "as (metric, value)")
)
print("Unpivoted:")
unpivoted.show()

# 4. Higher-order functions
print("\n=== HIGHER-ORDER FUNCTIONS ===")
df.withColumn("high_scores",
    F.filter("score_history",
              lambda x: x > 89)
).withColumn("scores_x2",
    F.transform("score_history",
                 lambda x: x * 2)
).withColumn("total_score",
    F.aggregate(
        "score_history",
        F.lit(0.0),
        lambda acc, x: acc + x
    )
).select(
    "model","high_scores",
    "scores_x2","total_score"
).show()

# 5. Spark SQL hints
print("\n=== QUERY HINTS ===")
print("""
Broadcast hint:
df.hint("broadcast")
spark.sql("SELECT /*+ BROADCAST(t) */ ...")

Repartition hint:
df.hint("repartition", 10)
df.hint("repartition", 10, "col")

Coalesce hint:
df.hint("coalesce", 5)

Skew hint (AQE):
df.hint("skew", "col")

These hints SUGGEST to optimizer
→ Optimizer may ignore if not beneficial
→ Explicit F.broadcast() is stronger!
""")

# 6. Spark SQL lateral view
print("\n=== LATERAL VIEW (explode) ===")
spark.sql("""
    SELECT model, score
    FROM df_view
    LATERAL VIEW explode(score_history) t AS score
""") if False else None  # demo syntax

print("""
SQL equivalent of explode():
SELECT model, score
FROM my_table
LATERAL VIEW explode(score_history) t AS score

Same as DataFrame:
df.withColumn("score",
    F.explode("score_history"))
""")

# Log to MLflow
mlflow.set_experiment("phase2_final_recap")
with mlflow.start_run(run_name="advanced_recap"):
    mlflow.log_param("topics", [
        "array_functions", "window_advanced",
        "pivot_unpivot", "higher_order",
        "query_hints", "lateral_view"
    ])
    mlflow.log_metric("phase2_days_complete", 25)
    print("\nPhase 2 recap logged!")

print("\n" + "="*60)
print("Phase 2 Advanced Recap COMPLETE!")
print("Ready for Phase 3! 🚀")
print("="*60)
