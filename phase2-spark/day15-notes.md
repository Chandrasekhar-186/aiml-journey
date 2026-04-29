## MLflow Monitoring Pattern
For each model in Gold table:
1. Start MLflow run with model name
2. Log all performance metrics
3. Log drift metrics + alert flag
4. Set tags: HEALTHY or DRIFT_DETECTED
5. Compare runs to find degradation

## Model Registry Lifecycle
None → Staging → Production → Archived
Staging: testing + validation
Production: serving live traffic
Archived: retired but preserved

## Duplicate Records Fix
Problem: job fails + retries = duplicates
Solution 1: idempotent MERGE
  DeltaTable.merge(new_data, "id = id")
            .whenNotMatchedInsertAll()
            .execute()
Solution 2: use batch_id in foreachBatch
  Check if batch_id already processed
Solution 3: Delta ACID = no partial writes
  But retried jobs can still duplicate!

## Top N per Group Pattern
from pyspark.sql.window import Window
window = Window.partitionBy("group_col") \
               .orderBy(F.desc("metric"))
df.withColumn("rank",
    F.rank().over(window)) \
  .filter(F.col("rank") <= N)

## 2D DP — Job Scheduling
dp[i][d] = min cost for i jobs in d days
Transition: try all possible last-day starts
            track max in last day's range
Key: dp[0][0] = 0, all others = INF initially

## Prefix Sum Pattern
prefix[i+1] = prefix[i] + nums[i]
Sum [l, r] = prefix[r+1] - prefix[l]
O(1) range sum after O(n) build!
