## Databricks SQL vs Spark SQL
DBSQL:    serverless, Photon, BI-optimized
          pay per query, auto-scales to zero
Spark SQL: general purpose, JVM, full control
           always-on cluster, ETL + streaming

## Photon Engine
C++ vectorized (not JVM like Tungsten)
SIMD: process 8+ values per instruction
5-10x faster for analytics aggregations
Enabled by default in Databricks SQL
Disabled for: Python UDFs, Pandas UDFs

## SQL Warehouse Types
Classic:    always-on, lower first-query latency
Serverless: starts in <5s, auto-scales to zero
Pro:        highest performance, best concurrency

## Pandas API on Spark
import pyspark.pandas as ps
Identical pandas syntax → runs on Spark!
Good for: existing pandas code, prototyping
Bad for:  complex joins, streaming, max performance

## Lakehouse Architecture (Bronze/Silver/Gold)
Bronze: raw data as-is (append only)
Silver: cleaned, validated, enriched
Gold:   business aggregates, ML features

## House Robber Pattern Variants
Linear (I):   dp with prev1, prev2
Circular (II): run linear twice, skip first/last
Tree (III):    post-order DFS, return (rob, skip)
Core insight:  at each position: rob OR skip
