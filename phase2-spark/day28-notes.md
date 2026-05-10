## Two Heaps Pattern — Median Stream
Lower half: max-heap (negate for Python!)
Upper half: min-heap
Invariant:  max(lower) <= min(upper)
Balance:    |lower| - |upper| <= 1

Steps for addNum(x):
1. Push to lower (max-heap)
2. If max(lower) > min(upper): rebalance
3. Balance sizes (lower can exceed by 1)

findMedian():
Equal size:   average of both tops
Lower larger: lower top is median

Applications:
→ Running median (streaming!)
→ Sliding window median
→ Spark percentile_approx internally!

## Phase 2 Superday Stories — Key Updates
Story 1: ML Monitor architecture (new!)
Story 2: Spark optimization 11x speedup (new!)
Story 3: Drift detection learning (new!)
Story 4: Feature store design (enhanced!)

All stories now include:
→ Specific Databricks components
→ Exact metrics (11x, <1 min, 7 competencies)
→ Production-grade decisions
→ Why choices were made

## Phase 3 Mindset
Phase 1: breadth (cover everything)
Phase 2: depth (Spark mastery)
Phase 3: math (understand from first principles)

Goal: when interviewer asks
"How does gradient descent work?"
→ NOT: "it minimizes the loss function"
→ YES: "it computes partial derivatives
        of J(θ) with respect to each θ_j,
        then steps in the negative gradient
        direction by learning rate α.
        Convergence guaranteed for convex J
        with appropriate α schedule."

THAT is Phase 3 depth.

## 2 Days Left — Phase 2 Final Checklist
□ GitHub: 58 green squares visible
□ Project: complete + documented
□ Certs: 2 free earned
□ Resume v4: updated
□ Referrals: 10 engineers contacted
□ LeetCode: 113+ solved
□ Cert mocks: 83-88% consistent
□ Phase 3: folder ready
