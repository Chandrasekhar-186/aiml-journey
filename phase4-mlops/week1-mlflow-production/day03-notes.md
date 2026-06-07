## Databricks Workflows Key Concepts
Job:     complete pipeline
Task:    single unit (notebook/Python/JAR)
Cluster: job cluster (clean) vs interactive
Depends: task_key dependencies = DAG!
Trigger: schedule (cron) or manual

## vs Airflow
Airflow:    general, external, complex setup
Workflows:  Databricks-native, simpler
            tight Unity Catalog integration
            Git-backed notebooks

## Delta Live Tables (DLT)
@dlt.table:              declare output table
@dlt.expect:             log quality failures
@dlt.expect_or_drop:     drop bad rows
@dlt.expect_or_fail:     stop on failure
dlt.read():              batch dependency
dlt.read_stream():       streaming dependency

Modes:
Triggered:  run on schedule (batch)
Continuous: always running (streaming)

## Autoloader (cloudFiles)
.format("cloudFiles")
.option("cloudFiles.format", "json")
.option("cloudFiles.schemaLocation", path)
→ Detects new files automatically
→ Exactly-once processing
→ Scales to millions of files!
→ Schema inference + evolution built-in

## Workflow Conditional Logic
condition_task: branch on task output value
{{tasks.task_key.values.metric}} > threshold
→ Conditional promotion!

## Counting Bits Pattern
dp[i] = dp[i >> 1] + (i & 1)
i >> 1: drop last bit (= i // 2)
i & 1:  is last bit 1? (= i % 2)
Reuses previously computed answers!
O(n) solution, no Math.popcount needed
