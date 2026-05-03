# Speed drill — 8 minutes max!
def longestCommonSubsequence(t1, t2):
    m, n = len(t1), len(t2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if t1[i-1] == t2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j],
                                dp[i][j-1])
    return dp[m][n]
print(longestCommonSubsequence(
    "abcde","ace"))  # 3


# Phase 2 Day 21 — LeetCode #1268
# Search Suggestions System
# Difficulty: Medium ⚡
# Approach: Sort + binary search
# Time: O(n log n + m log n) | Space: O(n)

import bisect

def suggestedProducts(products, searchWord):
    products.sort()
    result = []
    prefix = ""

    for char in searchWord:
        prefix += char
        # Find insertion point for prefix
        pos = bisect.bisect_left(products, prefix)
        suggestions = []

        # Check next 3 products after insertion point
        for i in range(pos, min(pos+3,
                                  len(products))):
            if products[i].startswith(prefix):
                suggestions.append(products[i])
            else:
                break

        result.append(suggestions)

    return result

print(suggestedProducts(
    ["mobile","mouse","moneypot",
     "monitor","mousepad"],
    "mouse"))
# [["mobile","moneypot","monitor"],
#  ["mobile","moneypot","monitor"],
#  ["mouse","mousepad"],
#  ["mouse","mousepad"],
#  ["mouse","mousepad"]]
