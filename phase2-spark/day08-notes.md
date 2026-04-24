## Liquid Clustering vs Z-ORDER vs Partitioning

Partitioning:
→ Physical folders per value
→ Best: low cardinality (date, region)
→ Fast: queries filtering on partition col
→ Problem: too many small files if high cardinality

Z-ORDER:
→ Co-locates data by column within files
→ Best: medium-high cardinality
→ Must rewrite table to re-cluster
→ Static — can't change columns easily

Liquid Clustering:
→ Incremental clustering (new!)
→ Best: high cardinality, mixed workloads
→ Can change columns anytime
→ Automatic background optimization

## Change Data Feed — 4 change types
insert:          new row added
update_preimage:  row before update
update_postimage: row after update
delete:          row removed

## Unity Catalog — 3 level namespace
catalog.schema.table
ml_prod.experiments.results

## Kafka Key Options
startingOffsets: "earliest" or "latest"
checkpointLocation: ALWAYS required!
failOnDataLoss: false for prod
maxOffsetsPerTrigger: rate limiting

## Graph with Weights — BFS pattern
Build bidirectional weighted graph
BFS tracks accumulated product/sum
Return -1 if not reachable

## Prim's MST — Greedy
Start from any node
Always pick cheapest edge to unvisited node
Use min-heap for efficient selection
Stop when all nodes visited
