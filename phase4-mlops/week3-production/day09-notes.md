## Three Pillars of ML Observability
Logs:    WHAT happened (structured JSON!)
Metrics: HOW MUCH/FAST (Prometheus)
Traces:  WHY it happened (MLflow spans)

## Structured Log Schema
{timestamp, level, service, model_version,
 event, request_id, latency_ms, confidence}
→ Queryable in Databricks Log Analytics
→ JSON format: always use, never print()

## SLA Thresholds (industry standard)
p99 latency: < 200ms (online serving)
Error rate:  < 1% (production)
Accuracy:    > 85% (model quality)
Drift:       PSI < 0.2 (stable)

## Cost Optimization Hierarchy
1. Right-size cluster (biggest impact!)
2. Spot instances (70-90% savings)
3. Auto-terminate (no idle waste)
4. Partition pruning (less data read)
5. Delta caching (avoid S3 re-reads)
6. Photon engine (10x SQL speedup)
7. Serverless (pay per second)

## Jump Game Greedy Pattern
Track max_reach = farthest reachable index
For each i: if i > max_reach → stuck!
Jump II: treat as BFS levels
curr_end = current "level" boundary
When i == curr_end → must jump!
jumps++ when crossing level boundary

## Phase 4 Week 3 Plan
Day 9:  Observability + cost     ✅ today
Day 10: Advanced streaming       ← tomorrow
Day 11: Phase 4 capstone project
Day 12: Phase 4 complete + review
