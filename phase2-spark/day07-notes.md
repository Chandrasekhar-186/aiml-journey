## Delta Lake Transaction Log — Key Facts

_delta_log/ folder = write-ahead log
Each commit = one JSON file
Checkpoint every 10 versions = parquet snapshot
Reading table = replay log from last checkpoint

## ACID via Transaction Log
Atomicity:   JSON written atomically or not at all
Consistency: Schema checked on every write
Isolation:   Optimistic concurrency —
             last writer wins on conflict
Durability:  Log written BEFORE data files

## Delta Operations → Versions
INSERT/append  → version +1
UPDATE         → version +1 (copy-on-write)
DELETE         → version +1 (soft delete)
MERGE          → version +1 (combined)
OPTIMIZE       → version +1 (compaction)
Schema change  → version +1 (metadata)

## Copy-on-Write vs Merge-on-Read
Copy-on-Write (Delta default):
→ Rewrite entire file on update
→ Fast reads, slow writes
→ Good for read-heavy workloads

Merge-on-Read (Delta optional):
→ Write delete vectors, apply at read time
→ Fast writes, slightly slower reads
→ Enable: delta.enableDeletionVectors=true

## State Machine DP Template
states = initial values
for each item:
    new_states based on transitions
    update all states simultaneously!
Key: use prev_ variables to avoid
     using updated state in same step!

## BFS for Shortest Path
Always BFS (not DFS!) for shortest path
BFS guarantees minimum steps
Each level = one more step from source
