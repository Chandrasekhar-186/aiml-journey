## System Design — Databricks Answer Framework

Always mention these Databricks tools:
Storage:    Delta Lake (not S3/HDFS)
Compute:    Apache Spark (not MapReduce)
ML Track:   MLflow (not custom logging)
Serving:    MLflow Model Serving
Streaming:  Spark Structured Streaming
Governance: Unity Catalog

## Lakehouse = Data Warehouse + Data Lake
Data Lake:      cheap storage, no structure
Data Warehouse: structured, expensive, no ML
Lakehouse:      cheap + structured + ML ready
                → Delta Lake achieves this!

## YOLOv8 Architecture
Backbone → feature extraction
Neck     → multi-scale feature fusion
Head     → bbox + class prediction
NMS      → remove duplicate detections

## Topological Sort — When to use
Use when: directed graph, check for cycles,
          find valid ordering of tasks
Examples: course prerequisites, build systems,
          Spark DAG execution order!

## Backtracking on Grid
def dfs(r, c, state):
    if complete: return True
    mark_visited(r, c)        # temp = board[r][c]
    board[r][c] = '#'         # mark
    for dr, dc in directions:
        if dfs(r+dr, c+dc, ...): return True
    board[r][c] = temp        # RESTORE!
    return False
