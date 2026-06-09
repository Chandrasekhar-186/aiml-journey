## Docker Key Concepts
Image:     immutable blueprint (layers)
Container: running image instance
Layer:     each RUN/COPY instruction
Registry:  Docker Hub, AWS ECR, GCR

## Dockerfile Best Practices for ML
1. Start with slim base (python:3.10-slim)
2. COPY requirements FIRST (layer cache!)
3. RUN pip install (cached if no req change)
4. COPY code last (changes most often)
5. HEALTHCHECK for orchestrator
6. Multi-stage build for smaller images

## FastAPI for ML
@app.on_event("startup"): load model ONCE
Pydantic: automatic request validation
Async: handles concurrent requests
Prometheus: metrics collection built-in
/health: required for k8s liveness probe

## Critical: Load Model at Startup!
WRONG: load model inside predict()
       → reload every request → 10x slower!
RIGHT: load at startup, reuse every request
       → 1 load, N fast predictions

## Coin Change DP Pattern
dp[0] = 0 (base case)
dp[i] = min(dp[i-coin]+1) for all coins
Build from 0 to amount
Return dp[amount] or -1

## 90 Day Stats 🏆
Phase 1: ✅ Foundations
Phase 2: ✅ Spark mastery
Phase 3: ✅ ML+DL+CV+LLMs
Phase 4: 6/30 underway
LeetCode: 178+
Projects: 5 complete
Streak:   90 perfect days!
