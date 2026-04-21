# Phase 2 Day 5 — LeetCode #5 Longest Palindrome
# Difficulty: Medium ⚡
# Approach: Expand around center
# Time: O(n²) | Space: O(1)

def longestPalindrome(s):
    result = ""
    result_len = 0

    def expand(left, right):
        nonlocal result, result_len
        while (left >= 0 and
               right < len(s) and
               s[left] == s[right]):
            if right - left + 1 > result_len:
                result = s[left:right+1]
                result_len = right - left + 1
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)      # odd length
        expand(i, i + 1)  # even length

    return result

print(longestPalindrome("babad"))   # "bab"
print(longestPalindrome("cbbd"))    # "bb"
print(longestPalindrome("racecar")) # "racecar"


# Phase 2 Day 5 — LeetCode #72 Edit Distance
# Difficulty: Hard 🔴
# Approach: 2D DP (classic!)
# Time: O(m*n) | Space: O(m*n)

def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    # dp[i][j] = edits to convert
    #            word1[:i] to word2[:j]
    dp = [[0]*(n+1) for _ in range(m+1)]

    # Base cases
    for i in range(m+1): dp[i][0] = i  # delete all
    for j in range(n+1): dp[0][j] = j  # insert all

    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # no edit!
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )

    return dp[m][n]

print(minDistance("horse", "ros"))    # 3
print(minDistance("intention","execution")) # 5
print(minDistance("", "abc"))         # 3
