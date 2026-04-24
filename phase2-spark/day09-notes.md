## Streaming Window Types
Tumbling:  fixed, non-overlapping
           window("ts", "1 minute")
Sliding:   fixed, overlapping
           window("ts", "2 min", "30 sec")
Session:   dynamic, based on activity gap
           session_window("ts", "5 minutes")

## Watermark Formula
.withWatermark("event_time", "10 minutes")

State kept for events where:
event_time >= max_seen_event_time - 10 minutes

Late data handling:
→ Within watermark: included in window
→ Beyond watermark: dropped

## foreachBatch — Most Powerful Sink
def process_batch(df, batch_id):
    # Full DataFrame API here!
    # Write to multiple sinks
    # Call MLflow, external APIs
    # Complex business logic

.writeStream.foreachBatch(process_batch)

## Streaming Checkpointing
Stores: offsets + state + metadata
Location: reliable storage (HDFS/S3/DBFS)
Enables: exactly-once processing
Required: ALWAYS set in production!

## Minimax Path (Swim in Water)
Instead of min(sum of costs):
Find min(max of costs along path)
Use Dijkstra with max() instead of sum()
heap = [(grid[0][0], 0, 0)]
cost = max(current_cost, neighbor_value)

## In-degree / Out-degree Pattern
Build degree arrays for graph problems
Judge: out_degree = 0, in_degree = n-1
Celebrity: same pattern!
Works for: trust problems, influence
