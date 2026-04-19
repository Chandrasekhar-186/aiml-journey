## Spark Memory — 3 Regions
Reserved (300MB): Spark internal use
User (40%):       Your code objects
Unified (60%):    Execution + Storage shared

## Storage Levels for Cert
MEMORY_ONLY:          fast, uses most memory
MEMORY_AND_DISK:      most common in prod!
MEMORY_ONLY_SER:      compact, slower CPU
DISK_ONLY:            slow, minimal memory
OFF_HEAP:             no GC, Tungsten native

## Shuffle Triggers (memorize!)
TRIGGERS:    groupBy, join, distinct,
             repartition, orderBy
NO SHUFFLE:  filter, select, withColumn,
             union, coalesce, map

## Cache vs Persist
cache()  = persist(MEMORY_ONLY)
persist(MEMORY_AND_DISK) = production choice
Always unpersist() when done!

## Dijkstra Template
dist = {node: inf for all nodes}
dist[src] = 0
heap = [(0, src)]
while heap:
    d, node = heappop(heap)
    if d > dist[node]: continue  # stale!
    for neighbor, weight in graph[node]:
        if dist[node] + weight < dist[neighbor]:
            dist[neighbor] = dist[node] + weight
            heappush(heap, (dist[neighbor], neighbor))

## Broadcast Variables vs Broadcast Join
Broadcast variable: share read-only data
                    with all executors
Broadcast join:     replicate small table
                    to avoid shuffle
Both use same mechanism internally!
