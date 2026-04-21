## Partitioning Rules — Memorize for Cert!
repartition(n):        full shuffle, any n
repartition(n, "col"): shuffle by column hash
coalesce(n):           reduce only, no shuffle
bucketBy(n, "col"):    pre-partition for joins

## Bucketing = Pre-partitioned Join
Two bucketed tables on same key + same n:
→ JOIN requires NO shuffle!
→ Spark knows data already co-located
→ Massive performance win for repeated joins

## Partition Count Formula
Shuffle partitions: dataset_size_MB / 128
Reading partitions: handled by maxPartitionBytes
Output partitions:  2-4 × CPU cores
AQE:                automates all of this!

## Streaming Output Modes
append:   new rows only (no aggregation)
complete: all rows (with aggregation)
update:   changed rows only (with aggregation)

## Watermark Formula
.withWatermark("timestamp", "10 minutes")
Means: accept events up to 10 min late
State cleanup: remove windows older than
               max_event_time - 10 minutes

## Edit Distance DP
if chars match:  dp[i][j] = dp[i-1][j-1]
if different:    dp[i][j] = 1 + min(
                   dp[i-1][j],   # delete
                   dp[i][j-1],   # insert
                   dp[i-1][j-1]  # replace
                 )
Base: dp[i][0] = i, dp[0][j] = j
