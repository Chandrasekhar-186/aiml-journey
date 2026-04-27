## GraphFrames Key Algorithms
PageRank:            node importance by connections
BFS:                 shortest path between nodes
Connected Components: isolated cluster detection
Triangle Count:      network density measure
Label Propagation:   community detection
Shortest Paths:      distance to landmark nodes

## GraphFrames vs GraphX
GraphFrames: DataFrame-based (Python ✅)
             Catalyst optimized ✅
             Easier API ✅
GraphX:      RDD-based (Scala/Java only)
             No Python support ❌
             Lower-level API ❌
→ Always use GraphFrames in Python!

## Production Pattern Summary
1. Always define schema explicitly
2. Handle bad records with mode option
3. Use dynamic partition overwrite
4. Make writes idempotent with MERGE
5. Monitor active streaming queries

## Reverse DFS/BFS Pattern
Instead of finding surrounded/captured:
Find SAFE (border-connected) first
Mark safe with temporary marker
Then flip remaining = captured
Works for: surrounded regions,
           enclosed islands, safe zones

## Streaming Exactly-Once
Requirement: idempotent sink
Delta Lake:  ACID = exactly-once ✅
Kafka:       requires idempotent producer
Custom:      use batch_id for deduplication
