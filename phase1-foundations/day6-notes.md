## Docker — Plain English
Container = lightweight VM that packages your
code + dependencies + environment together.

Dockerfile → Image → Container
(recipe)     (cake)   (eating the cake)

Why Databricks uses Docker:
→ Reproducible ML experiments
→ Same environment dev → staging → production
→ Easy scaling on Kubernetes

## Statistics Quick Reference
Mean     = average (sensitive to outliers)
Median   = middle value (robust to outliers)
Variance = average squared deviation from mean
Std Dev  = sqrt(variance) — same unit as data
68-95-99.7 rule: 1,2,3 std devs cover
                 68%, 95%, 99.7% of normal data

## Delta Lake — 3 Superpowers
1. ACID transactions — no corrupt data
2. Time travel — query any historical version
3. Schema enforcement — no bad data sneaks in

## Tree Recursion Pattern:
Base case:  if not node → return 0/True/None
Recursive:  solve(left) + solve(right)
Combine:    return result based on both
