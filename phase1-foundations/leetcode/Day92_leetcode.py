# Phase 4 Day 8 — LeetCode #300
# Longest Increasing Subsequence
# Difficulty: Medium ⚡
# Approach: DP O(n²) or binary search O(n log n)
# Time: O(n log n) | Space: O(n)

import bisect

def lengthOfLIS(nums):
    # tails[i] = smallest tail of all
    # increasing subseq of length i+1
    tails = []
    for n in nums:
        pos = bisect.bisect_left(tails, n)
        if pos == len(tails):
            tails.append(n)
        else:
            tails[pos] = n  # replace!
    return len(tails)

print(lengthOfLIS([10,9,2,5,3,7,101,18])) # 4
print(lengthOfLIS([0,1,0,3,2,3]))          # 4
print(lengthOfLIS([7,7,7,7,7,7,7]))        # 1

# Speed — 5 min! DP decode
def numDecodings(s):
    if not s or s[0] == '0': return 0
    n = len(s)
    dp = [0] * (n+1)
    dp[0] = dp[1] = 1
    for i in range(2, n+1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        two = int(s[i-2:i])
        if 10 <= two <= 26:
            dp[i] += dp[i-2]
    return dp[n]

print(numDecodings("12"))   # 2
print(numDecodings("226"))  # 3
print(numDecodings("06"))   # 0
