## Cert Simulation 2 — Key Additions

SparkSession vs SparkContext:
SparkSession = unified entry (Spark 2.0+)
SparkContext = accessible via spark.sparkContext

ROWS BETWEEN vs RANGE BETWEEN:
ROWS: count physical rows
      ROWS BETWEEN -2 AND 0 = last 3 rows
RANGE: value-based range
       RANGE BETWEEN -10 AND 0 = values within 10

pivot() syntax:
df.groupBy("row").pivot("col").agg(F.sum("val"))
→ Creates one column per unique "col" value

monotonically_increasing_id():
→ Unique but NOT consecutive!
→ Has gaps between partitions
→ Use for: unique IDs, not for ordering!

CONVERT TO DELTA:
→ Adds _delta_log to existing Parquet
→ No data rewrite = very fast!
→ Then can use all Delta features

## Topological Sort Template (BFS/Kahn's)
Build graph + indegree array
Initialize queue with all 0-indegree nodes
While queue:
    pop node → add to order
    for each neighbor: indegree -= 1
    if indegree == 0: add to queue
If order length == n: valid order
Else: cycle detected!

## Phase 2 Final 13 Days
Day 48: Cert sim 2 ← today
Day 49: Gap fill from sim 2
Day 50: Mock Spark interview Round 3
Day 51: Phase 2 project final polish
Day 52: Advanced topics recap
Day 53-58: Final cert revision daily
Day 59-61: EXAM + Phase 3 launch!
