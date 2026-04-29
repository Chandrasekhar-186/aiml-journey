# Phase 2 Day 15 — LeetCode #1335
# Min Difficulty Job Schedule
# Difficulty: Hard 🔴
# Approach: 2D DP
# Time: O(n²*d) | Space: O(n*d)

def minDifficulty(jobDifficulty, d):
    n = len(jobDifficulty)
    if n < d:
        return -1  # can't schedule!

    # dp[i][j] = min difficulty scheduling
    #            first i jobs in j days
    INF = float('inf')
    dp = [[INF] * (d+1) for _ in range(n+1)]
    dp[0][0] = 0

    for i in range(1, n+1):
        for day in range(1, min(i, d)+1):
            # Try all possible last-day starts
            max_diff = 0
            for k in range(i, day-1, -1):
                max_diff = max(
                    max_diff,
                    jobDifficulty[k-1]
                )
                if dp[k-1][day-1] < INF:
                    dp[i][day] = min(
                        dp[i][day],
                        dp[k-1][day-1] + max_diff
                    )

    return dp[n][d] if dp[n][d] < INF else -1

print(minDifficulty([6,5,4,3,2,1], 2))  # 7
print(minDifficulty([9,9,9], 4))         # -1
print(minDifficulty([1,1,1], 3))         # 3



# Phase 2 Day 15 — LeetCode #2090
# K Radius Subarray Averages
# Difficulty: Medium ⚡
# Approach: Sliding window prefix sum
# Time: O(n) | Space: O(n)

def getAverages(nums, k):
    n = len(nums)
    result = [-1] * n
    window = 2 * k + 1

    if window > n:
        return result

    # Build prefix sum
    prefix = [0] * (n + 1)
    for i, num in enumerate(nums):
        prefix[i+1] = prefix[i] + num

    for i in range(k, n-k):
        total = prefix[i+k+1] - prefix[i-k]
        result[i] = total // window

    return result

print(getAverages([7,4,3,9,1,8,5,2,6], 3))
# [-1,-1,-1,5,4,4,-1,-1,-1]
print(getAverages([100000], 0))
# [100000]
