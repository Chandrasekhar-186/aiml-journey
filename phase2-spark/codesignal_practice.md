# CodeSignal OA Practice — Day 53
Date: May 3, 2026
Format: 4 problems, 70 minutes
Target score: 850+ (Databricks requirement)

## CODESIGNAL FORMAT
Problem 1: Easy/Medium — 15 min
Problem 2: Medium     — 20 min
Problem 3: Medium/Hard — 20 min
Problem 4: Hard        — 15 min
(time allocation is flexible)

## KEY DIFFERENCES from LeetCode
1. Score based on: correctness + performance
   → Partial credit for passing some test cases
   → TLE (timeout) = partial score, not zero!

2. Pre-written boilerplate — you fill the function

3. Auto-tests run instantly — fast feedback

4. No hints, no editorial during exam

5. Score = weighted combination of test cases
   → 850+ = need most Medium correct + some Hard

## STRATEGY FOR 850+

First 5 minutes:
→ Read ALL 4 problems quickly
→ Identify the easy ones
→ Decide order (easiest → hardest)

During solving:
→ Brute force FIRST (gets partial credit!)
→ Optimize if time allows
→ Leave working brute force over broken optimal

Test case strategy:
→ Always test edge cases: [], [1], duplicates
→ Empty string, single element, all same
→ Large input (check for TLE mentally)

Time management:
→ Never spend >25 min on one problem
→ Move on → return if time remains
→ Incomplete brute force > no solution

## PRACTICE PROBLEMS — Do these timed!

Set timer: 70 minutes. Solve all 4:

### Problem 1 (Easy — 15 min target)
Given array nums and target k,
return number of subarrays with sum = k.

def countSubarrays(nums, k):
    # Use prefix sum + hashmap
    from collections import defaultdict
    count = defaultdict(int)
    count[0] = 1
    prefix = result = 0
    for n in nums:
        prefix += n
        result += count[prefix - k]
        count[prefix] += 1
    return result

Test: countSubarrays([1,1,1], 2) = 2
      countSubarrays([1,2,3], 3) = 2

### Problem 2 (Medium — 20 min target)
Given a matrix, find longest increasing path.

def longestIncreasingPath(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(r, c):
        if (r, c) in memo: return memo[(r,c)]
        best = 1
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if (0<=nr<m and 0<=nc<n and
                    matrix[nr][nc] > matrix[r][c]):
                best = max(best, 1 + dfs(nr, nc))
        memo[(r,c)] = best
        return best

    return max(dfs(r,c) for r in range(m)
               for c in range(n))

Test: longestIncreasingPath([[9,9,4],[6,6,8],[2,1,1]]) = 4

### Problem 3 (Medium/Hard — 20 min target)
Given n, return all valid combinations of
n pairs of parentheses.

def generateParenthesis(n):
    result = []
    def backtrack(s, open, close):
        if len(s) == 2*n:
            result.append(s)
            return
        if open < n:
            backtrack(s+'(', open+1, close)
        if close < open:
            backtrack(s+')', open, close+1)
    backtrack("", 0, 0)
    return result

Test: generateParenthesis(3) = 5 combinations

### Problem 4 (Hard — 15 min target)
Given array, find maximum sum of
non-adjacent elements in a circle.

def maxSumCircular(nums):
    def kadane(arr):
        cur = res = arr[0]
        for n in arr[1:]:
            cur = max(n, cur + n)
            res = max(res, cur)
        return res

    if max(nums) < 0: return max(nums)
    total = sum(nums)
    # Case 1: max subarray (not circular)
    # Case 2: total - min subarray (circular)
    return max(kadane(nums),
               total - kadane([-n for n in nums]))

Test: maxSumCircular([1,-2,3,-2]) = 3
      maxSumCircular([5,-3,5]) = 10

## SCORING ESTIMATE
P1 (Easy):   solved correctly → +250 pts
P2 (Medium): solved correctly → +250 pts
P3 (Medium): solved correctly → +200 pts
P4 (Hard):   partial solve    → +150 pts
Total estimate: 850 pts ← TARGET!

## AFTER PRACTICE
→ Note which patterns felt slow
→ Those patterns = drill tomorrow
→ Target: all 4 solved in <60 min
