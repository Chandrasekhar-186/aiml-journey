## Week 2 Project Deliverables
FastAPI: full prod grade with middleware
Docker: Dockerfile + Compose + health
Prometheus: 5 metrics (counter+histogram+gauge)
Grafana: 5 dashboard panels
Batch: Spark UDF → Delta predictions
MLflow: model registered + benchmarked

## Prometheus Metric Types
Counter:   monotonically increasing (requests)
Histogram: distribution + percentiles (latency)
Gauge:     current value up or down (accuracy)
Summary:   like histogram, client-side

## Key FastAPI Patterns
lifespan context manager: startup/shutdown
@app.middleware("http"): intercept all
Request state: track across middleware
uuid4(): unique request IDs for tracing

## LIS Binary Search O(n log n)
tails array: smallest tail per length
bisect_left: find insertion position
pos==len: new longest → append
pos<len:  can replace with smaller tail
Never explicitly stores the subsequence!

## Decode Ways DP
dp[i]: ways to decode s[:i]
If s[i-1]!='0': dp[i] += dp[i-1]
If 10<=s[i-2:i]<=26: dp[i] += dp[i-2]
Base: dp[0]=1 (empty), dp[1]=1 if s[0]!='0'

## Phase 4 Week 2 Complete
Day 6:  Docker + FastAPI      ✅
Day 7:  Model Serving         ✅
Day 8:  Week 2 Project        ✅ today
→ Week 3: Production patterns
→ FastAPI advanced + streaming
