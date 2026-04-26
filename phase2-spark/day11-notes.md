## Spark Cert — Most Commonly Missed

1. collect() = ACTION not transformation
2. coalesce() reduces WITHOUT shuffle
   repartition() can increase WITH shuffle
3. RANK: 1,2,2,4 | DENSE_RANK: 1,2,2,3
   ROW_NUMBER: always 1,2,3,4
4. WHERE = pre-group | HAVING = post-group
5. cache() = MEMORY_ONLY (not MEMORY_AND_DISK!)
6. Checkpoint every 10 Delta versions
7. Watermark BOUNDS STATE (not just filters!)
8. foreachBatch = full DF API per batch
9. reduceByKey = map-side combine first!
10. AQE needs Spark 3.0+ (default in 3.2+)

## Multi-source BFS Pattern
Start BFS from ALL sources simultaneously
→ Walls and Gates: start from all gates (0)
→ Rotting Oranges: start from all rotten (2)
→ Pacific Atlantic: start from all borders
Key: add ALL sources to queue BEFORE starting!

## Union-Find — countComponents
Start with n components
Each successful union → components -= 1
Final answer = remaining components
