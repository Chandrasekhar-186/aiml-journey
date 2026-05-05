## CodeSignal Strategy — Key Rules
1. Read ALL 4 problems first (5 min)
2. Solve easiest → hardest
3. Brute force first (partial credit!)
4. Never >25 min on one problem
5. Test edge cases before submitting
6. 850+ = most Medium + some Hard

## Score Distribution (estimated)
Easy correct:    250 pts
Medium correct:  250 pts
Medium correct:  200 pts
Hard partial:    150 pts
Total:           850 pts ← Databricks threshold

## Common CodeSignal Patterns 
1. Prefix sum + HashMap (subarray sum)
2. DFS + memo (matrix paths)
3. Backtracking (combinations, permutations)
4. Sliding window (substring problems)
5. Heap (top K problems)
6. DP (optimization problems)
7. Graph BFS/DFS (connectivity)

## DFS + Memo Pattern
memo = {}
def dfs(state):
    if state in memo: return memo[state]
    result = base_case
    for next_state in neighbors(state):
        result = combine(result, dfs(next_state))
    memo[state] = result
    return result

## Circular Subarray Max Sum
Case 1: Max subarray not wrapping
        → standard Kadane's
Case 2: Max subarray wrapping around
        → total - min subarray
        → min subarray = Kadane's on negated!
Answer: max(Case1, Case2)
Edge: if all negative → return max element

## GraphFrames Interview Points
Always use GraphFrames (not GraphX) in Python
Requires checkpoint for connected components
PageRank: reset=0.15, maxIter=10 typical
Applications: fraud, recommendation, knowledge
