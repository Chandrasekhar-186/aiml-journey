# System Design — Databricks Interview Prep
Date: March 30, 2026

## Core Concepts

### 1. Scalability
Vertical scaling:   add more power to one machine
Horizontal scaling: add more machines (preferred!)
Databricks uses:    horizontal scaling via Spark clusters

### 2. Load Balancing
Distributes traffic across multiple servers
Types: Round-robin, Least connections, IP hash
Databricks: distributes Spark tasks across workers

### 3. Caching
Store frequently accessed data in fast storage
L1/L2 cache → RAM → SSD → HDD → Network
Spark cache():    keeps DataFrame in cluster RAM
Delta Lake:       caches file metadata

### 4. CAP Theorem
Can only guarantee 2 of 3:
Consistency:  all nodes see same data
Availability: system always responds
Partition tolerance: survives network splits

Delta Lake choice: Consistency + Partition tolerance
→ ACID transactions over availability

### 5. Database Sharding
Split data across multiple databases
Spark does this automatically via partitions!
Delta Lake: data files = natural shards

## Databricks-Specific Architecture

### Lakehouse Architecture
Bronze layer: raw data (ingestion)
Silver layer: cleaned/filtered data
Gold layer:   business-level aggregates

### ML Pipeline at Scale
Data → Delta Lake (Bronze)
     → PySpark transform (Silver)
     → Feature engineering (Gold)
     → MLflow training
     → Model Registry
     → Model Serving
     → Monitoring + feedback loop

### Design Question: "Scale ML pipeline for 1TB/day"
1. Ingestion: Spark Structured Streaming
2. Storage:   Delta Lake with partitioning
3. Transform: PySpark with broadcast joins
4. Training:  Distributed ML on Databricks
5. Tracking:  MLflow experiment logging
6. Serving:   MLflow Model Serving + REST API
7. Monitor:   Delta Lake audit logs + alerts

## Key Numbers to Remember
Latency:
  L1 cache:     ~1ns
  RAM:          ~100ns
  SSD:          ~100μs
  Network:      ~1ms
  Databricks job: seconds to minutes

Throughput:
  Single machine: GBs/sec
  Spark cluster:  TBs/hour
  Delta Lake:     Petabyte scale

## System Design Template (4 steps)
1. Clarify requirements (2 min)
   - Scale? Users? Latency requirements?
2. High-level design (5 min)
   - Draw boxes: client → API → compute → storage
3. Deep dive (15 min)
   - Focus on bottlenecks + Databricks tools
4. Trade-offs (3 min)
   - What did you optimize for? What did you sacrifice?
```

**Practice answering out loud:**
```
"Design a real-time fraud detection ML system"

Your answer structure:
1. Clarify: "How many transactions/second?
            What latency is acceptable?
            Batch or real-time predictions?"
2. Design:  Kafka → Spark Streaming →
            Delta Lake → MLflow model →
            REST API → monitoring
3. Deep:    Explain each component's role
4. Tradeoff: "I chose Delta Lake over S3 because
              ACID transactions prevent corrupt
              fraud labels during concurrent writes"
