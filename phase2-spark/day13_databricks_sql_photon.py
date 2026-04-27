# Phase 2 Day 13 — Databricks SQL + Photon
# Date: April 23, 2026
# Databricks SQL = separate product, interview topic!

print("="*60)
print("Databricks SQL + Photon Engine")
print("="*60)

"""
DATABRICKS SQL OVERVIEW:

Databricks SQL (DBSQL) is a serverless
data warehouse built on Delta Lake.

Key differentiators vs traditional warehouses:
→ Runs on Delta Lake (open format!)
→ No data copying — query where data lives
→ Photon engine for 10x speedup
→ SQL endpoints (serverless or classic)
→ Built-in visualizations + dashboards
→ Integrates with BI tools (Tableau, PowerBI)
→ Unity Catalog governance built-in

Architecture:
SQL Warehouse → Photon Engine → Delta Lake
     ↑               ↑              ↑
  Serverless    C++ vectorized   ACID storage
  compute       execution        open format
"""

# 1. Databricks SQL vs Spark SQL
print("\n=== DATABRICKS SQL vs SPARK SQL ===")
print("""
Spark SQL:
→ General purpose (batch + streaming)
→ Scala/Python/Java/R APIs
→ JVM-based execution (Tungsten)
→ Lower-level control
→ Best for: ETL pipelines, ML preprocessing

Databricks SQL:
→ Optimized for BI/analytics queries
→ SQL-only (no Python API needed)
→ Photon execution engine (C++, not JVM!)
→ Serverless — auto-scales to zero
→ Best for: ad-hoc analytics, dashboards
→ 10-100x faster than Spark SQL on analytics!

When to use which:
ETL pipeline:     Spark + PySpark ✅
ML preprocessing: Spark + PySpark ✅
BI dashboard:     Databricks SQL ✅
Ad-hoc analysis:  Databricks SQL ✅
Streaming:        Spark Streaming ✅
""")

# 2. Photon Engine
print("\n=== PHOTON ENGINE ===")
print("""
Photon = Databricks' C++ vectorized engine

Why faster than Tungsten (Java/JVM):
→ C++ = no JVM overhead
→ SIMD vectorization (process 8+ values at once)
→ Cache-friendly memory layouts
→ Native support for modern CPU instructions

Performance gains:
→ Aggregations:  5-10x faster
→ Joins:         3-5x faster
→ Scan + filter: 2-4x faster

How to enable:
→ Databricks Runtime 9.1+
→ Automatically used by Databricks SQL
→ Configure: spark.databricks.photon.enabled=true

Photon compatible operations:
✅ Filter, select, aggregate, join
✅ Window functions
✅ Delta Lake reads/writes
✅ Spark SQL queries
❌ Python UDFs (falls back to Spark)
❌ Pandas UDFs (falls back to Spark)
""")

# 3. SQL Warehouse types
print("\n=== SQL WAREHOUSE TYPES ===")
print("""
Classic SQL Warehouse:
→ Always-on cluster
→ Lower latency for first query
→ Higher cost when idle
→ Best for: frequent, time-sensitive queries

Serverless SQL Warehouse:
→ Starts in <5 seconds
→ Auto-scales to zero when idle
→ Pay per query (not uptime)
→ Best for: sporadic analytics workloads
→ Managed by Databricks (no infra!)

Pro SQL Warehouse:
→ Highest performance
→ Photon enabled by default
→ Best concurrency for large teams
""")

# 4. Key Databricks SQL features
print("\n=== DATABRICKS SQL FEATURES ===")
print("""
Query History:
→ All queries logged automatically
→ Performance profiling built-in
→ Identify slow queries instantly

Query Profiles:
→ Visual execution plan
→ Time spent per operator
→ Data processed per stage
→ Similar to Spark UI but cleaner!

Alerts:
→ Set thresholds on query results
→ "Alert me if fraud_count > 100"
→ Slack/email/PagerDuty integration

Dashboards:
→ Live auto-refreshing dashboards
→ No Tableau license needed!
→ Share with stakeholders directly

Partner Connect:
→ One-click connection to:
   Tableau, PowerBI, Fivetran,
   dbt, Hightouch, Census
""")

# 5. Pandas API on Spark (Koalas)
print("\n=== PANDAS API ON SPARK ===")
print("""
Problem: pandas code doesn't scale
Solution: pandas API on Spark (formerly Koalas)

import pyspark.pandas as ps  # Spark 3.2+

# Looks exactly like pandas!
df = ps.read_csv("s3://my-bucket/data.csv")
df.groupby("model").accuracy.mean()
df.sort_values("score", ascending=False)
df.describe()

Under the hood: runs on Spark!
→ Distributed across cluster
→ No pandas limitations
→ Same API = zero code changes!

When to use:
✅ Existing pandas codebase
✅ Data scientists familiar with pandas
✅ Quick prototyping
✅ Medium-sized data (GBs not TBs)

When NOT to use:
❌ Need full Spark optimization
❌ Complex joins (use DataFrame API)
❌ Streaming (use Spark Streaming)
""")

# 6. Databricks SQL interview prep
print("\n=== INTERVIEW TALKING POINTS ===")
print("""
Q: "When would you use Databricks SQL
    vs PySpark for analytics?"

A: "For ad-hoc analytics and BI dashboards,
    Databricks SQL with Photon is 10x faster
    and serverless — pays for itself instantly.
    For ETL pipelines and ML preprocessing,
    PySpark gives full programmatic control.
    The beauty is both sit on top of Delta Lake,
    so the data is always consistent between
    the two access patterns."

Q: "What is Photon and why does it matter?"

A: "Photon is Databricks' C++ vectorized
    execution engine. Unlike Spark's Tungsten
    which generates JVM bytecode, Photon runs
    native C++ with SIMD vectorization —
    processing 8+ values per CPU instruction.
    For analytics workloads this gives 5-10x
    speedup on aggregations and joins."
""")
