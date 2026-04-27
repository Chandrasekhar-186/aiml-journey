# Streaming Complete Reference — Cert Prep
Date: April 22, 2026

## Output Modes — EXACT behavior

append:
→ Only NEW rows added since last trigger
→ Cannot use with aggregations
→ Use for: event logs, inserts only

complete:
→ ENTIRE result table every trigger
→ Must use with aggregations
→ Memory concern: result grows forever!
→ Use for: running totals, counts

update:
→ Only CHANGED rows since last trigger
→ Works with/without aggregations
→ Most efficient for large result sets
→ Use for: dashboard metrics

## Trigger Types — EXACT syntax

Default (process ASAP):
.trigger()  ← omit trigger entirely

Fixed interval:
.trigger(processingTime="30 seconds")

One-time batch:
.trigger(once=True)

Available now (Spark 3.3+):
.trigger(availableNow=True)

Continuous (experimental, low latency):
.trigger(continuous="1 second")

## Watermark — EXACT formula
.withWatermark("event_ts", "10 minutes")

Max event time seen: T
Watermark threshold: T - 10 minutes
Events with ts < (T - 10 min): DROPPED
State older than watermark: CLEANED UP

## Window Functions
Tumbling (non-overlapping):
window("ts", "1 minute")
→ [00:00-01:00], [01:00-02:00]...

Sliding (overlapping):
window("ts", "2 minutes", "30 seconds")
→ [00:00-02:00], [00:30-02:30]...

Session (gap-based, dynamic):
session_window("ts", "5 minutes")
→ Groups events < 5 min apart

## foreachBatch signature
def process(df: DataFrame, batch_id: int):
    # Full batch DataFrame API here!
    pass

.writeStream.foreachBatch(process)

## Fault Tolerance Guarantee
At-least-once:  default (with checkpoint)
Exactly-once:   with idempotent sink
                OR transactional sink
                (Delta Lake = exactly-once!)

## Streaming Anti-patterns
❌ No checkpoint = data loss on restart
❌ No watermark = OOM on aggregations
❌ collect() inside foreachBatch = OOM
❌ Python UDF in streaming = very slow
❌ Non-idempotent foreachBatch = duplicates
