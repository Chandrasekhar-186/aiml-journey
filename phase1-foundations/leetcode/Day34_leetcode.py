# Phase 2 Day 4 — LeetCode #518 Coin Change II
# Difficulty: Medium ⚡
# Approach: Unbounded knapsack DP
# Time: O(amount * coins) | Space: O(amount)

def change(amount, coins):
    # dp[i] = number of ways to make amount i
    dp = [0] * (amount + 1)
    dp[0] = 1  # one way to make 0: use nothing

    # Process each coin
    for coin in coins:
        for amt in range(coin, amount + 1):
            # Using this coin: ways to make
            # (amt - coin) + current ways
            dp[amt] += dp[amt - coin]

    return dp[amount]

# Test cases
print(change(5, [1,2,5]))    # 4
print(change(3, [2]))         # 0
print(change(10, [10]))       # 1


# Phase 2 Day 4 — LeetCode #1143 LCS
# Difficulty: Medium ⚡
# Approach: 2D Dynamic Programming
# Time: O(m*n) | Space: O(m*n)

def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    # dp[i][j] = LCS of text1[:i] and text2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                # Characters match — extend LCS
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # Take best of skip either char
                dp[i][j] = max(dp[i-1][j],
                                dp[i][j-1])

    return dp[m][n]

# Test cases
print(longestCommonSubsequence(
    "abcde", "ace"))    # 3
print(longestCommonSubsequence(
    "abc", "abc"))      # 3
print(longestCommonSubsequence(
    "abc", "def"))      # 0
