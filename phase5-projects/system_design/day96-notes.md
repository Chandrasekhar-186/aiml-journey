## System Design Interview Tips (Databricks)
Two types asked:
1. High-level: whiteboard architecture
2. Low-level: WRITE RUNNING CODE!

Low-level tips:
→ Start with data structures (classes)
→ Define interfaces first
→ Code the core operation fully
→ Add edge cases + error handling
→ Discuss scale: "for 1M users, I'd add Redis"

## Experiment Tracker Design
Data structures:
  ExperimentRun: params, metrics, tags, status
  Tracker: experiments dict, runs dict

Key operations:
  start_run → create + activate run
  log_param/metric/artifact → store data
  search_runs → filter + sort
  compare_runs → rank by metric

Scale considerations:
→ 1M+ runs: add indexing on experiment_id
→ Fast search: store metrics in sorted set
→ Artifacts: S3/DBFS paths, not content
→ Distributed: shard by experiment_id

## Rate Limiter Design
Sliding window: deque of timestamps
Per user: defaultdict(deque)
Allow if: len(queue) < max_requests
Evict: timestamps older than window

Other patterns:
Token bucket: tokens += rate, spend 1/req
Fixed window: counter per minute (simpler)
Sliding window: most accurate, use for ML API!

## Topological Sort Pattern
State: 0=unvisited 1=visiting 2=done
DFS: if visiting again → cycle!
Post-order append → reverse = topo order
Use for: task scheduling, dependency resolution
       = Databricks Workflows internally!

## Phase 5 Focus Areas
System design: low-level + high-level
Behavioral: 3 STAR stories polished
LeetCode: hard problems + speed drills
CodeSignal: 2 more full simulations
Superday sim 2: full mock interview
Referral: Round 4 outreach
