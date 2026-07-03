# Phase 4 Day 10 — LeetCode #152
# Maximum Product Subarray
# Difficulty: Medium ⚡
# Approach: Track max AND min (negatives flip!)
# Time: O(n) | Space: O(1)

def maxProduct(nums):
    result = max(nums)
    cur_min = cur_max = 1

    for n in nums:
        if n == 0:
            cur_min = cur_max = 1
            continue
        tmp = cur_max * n
        cur_max = max(n, cur_max*n, cur_min*n)
        cur_min = min(n, tmp, cur_min*n)
        result = max(result, cur_max)
    return result

print(maxProduct([2,3,-2,4]))    # 6
print(maxProduct([-2,0,-1]))     # 0
print(maxProduct([-2,3,-4]))     # 24

# Speed — 5 min! DP solved before
def wordBreak(s, wordDict):
    words = set(wordDict)
    dp = [False] * (len(s)+1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[len(s)]

print(wordBreak("leetcode",
    ["leet","code"]))    # True
print(wordBreak("applepenapple",
    ["apple","pen"]))    # True
