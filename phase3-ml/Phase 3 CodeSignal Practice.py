# Set timer: 70 minutes
# Solve all 4 — simulate exact OA conditions!

# P1 (15 min): Subarray Sum Equals K
from collections import defaultdict
def subarraySum(nums, k):
    count = defaultdict(int)
    count[0] = 1
    prefix = result = 0
    for n in nums:
        prefix += n
        result += count[prefix - k]
        count[prefix] += 1
    return result

# P2 (20 min): Longest Increasing Path
def longestIncreasingPath(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}
    def dfs(r, c):
        if (r,c) in memo: return memo[(r,c)]
        best = 1
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr,nc = r+dr, c+dc
            if (0<=nr<m and 0<=nc<n and
                    matrix[nr][nc]>matrix[r][c]):
                best = max(best, 1+dfs(nr,nc))
        memo[(r,c)] = best
        return best
    return max(dfs(r,c)
               for r in range(m)
               for c in range(n))

# P3 (20 min): Word Break
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    dp = [False] * (len(s)+1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]

# P4 (15 min): Max Circular Subarray Sum
def maxSumCircular(nums):
    def kadane(arr):
        cur = res = arr[0]
        for n in arr[1:]:
            cur = max(n, cur+n)
            res = max(res, cur)
        return res
    if max(nums) < 0: return max(nums)
    return max(kadane(nums),
               sum(nums) - kadane([-n for n in nums]))

# Run all tests!
print(subarraySum([1,1,1], 2))          # 2
print(longestIncreasingPath([[9,9,4],[6,6,8],[2,1,1]])) # 4
print(wordBreak("leetcode", ["leet","code"])) # True
print(maxSumCircular([1,-2,3,-2]))       # 3
