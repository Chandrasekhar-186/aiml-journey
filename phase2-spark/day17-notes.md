## Advanced Streaming Patterns Summary

Stream-static join:
→ Static side loaded ONCE at startup
→ No shuffle, very fast
→ Good for: model metadata, config
→ Stale if static table changes!

mapGroupsWithState:
→ Arbitrary state per group key
→ More powerful than window aggregations
→ Complex event detection, sessions
→ Harder to implement correctly

Deduplication:
→ dropDuplicates(["id", "ts"])
→ REQUIRES watermark for streaming!
→ Dedup window = watermark duration
→ Solves Kafka at-least-once problem

Delta as streaming source:
→ readStream.format("delta")
→ Reads only new committed data
→ ACID guarantees on reads
→ No Kafka needed for internal pipelines!

## Speed Drill Performance Targets
Word Break:    < 5 minutes
Coin Change:   < 5 minutes
LIS:           < 5 minutes
Rotting Oranges: < 8 minutes
Two Sum:       < 3 minutes
Valid Parens:  < 3 minutes

## Cert Speed — 60Q in 120 min
= 2 minutes per question
Speed drill builds this reflex!

## Phase 2 Remaining (14 days)
Days 47-50: Advanced patterns + cert prep
Days 51-54: Mock interviews + gap filling
Days 55-58: Final cert revision
Days 59-61: EXAM + Phase 3 prep
