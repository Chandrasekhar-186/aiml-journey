# Speed challenge — 10 min max!
from collections import Counter

def findAnagrams(s, p):
    if len(p) > len(s): return []
    p_count = Counter(p)
    window = Counter(s[:len(p)])
    result = [0] if window == p_count else []

    for i in range(len(p), len(s)):
        window[s[i]] += 1
        old = s[i-len(p)]
        window[old] -= 1
        if window[old] == 0:
            del window[old]
        if window == p_count:
            result.append(i-len(p)+1)
    return result

print(findAnagrams("cbaebabacd","abc"))  # [0,6]


# Phase 2 Day 26 — LeetCode #2466
# Count Ways to Build Good Strings
# Difficulty: Medium ⚡
# Approach: DP bottom-up
# Time: O(high) | Space: O(high)

def countGoodStrings(low, high, zero, one):
    MOD = 10**9 + 7
    # dp[i] = ways to build string of length i
    dp = [0] * (high + 1)
    dp[0] = 1  # empty string = 1 way

    for i in range(1, high + 1):
        if i >= zero:
            dp[i] = (dp[i] + dp[i-zero]) % MOD
        if i >= one:
            dp[i] = (dp[i] + dp[i-one]) % MOD

    return sum(dp[low:high+1]) % MOD

print(countGoodStrings(3, 3, 1, 1))   # 8
print(countGoodStrings(2, 3, 1, 2))   # 5
