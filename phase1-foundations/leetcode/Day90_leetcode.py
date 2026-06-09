# Speed — 3 min max! Fibonacci pattern
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b

print(climbStairs(2))  # 2
print(climbStairs(3))  # 3
print(climbStairs(5))  # 8

# Phase 4 Day 6 — LeetCode #322
# Coin Change
# Difficulty: Medium ⚡
# Approach: DP bottom-up
# Time: O(amount * coins) | Space: O(amount)

def coinChange(coins, amount):
    # dp[i] = min coins to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # base case

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i],
                             dp[i-coin] + 1)

    return dp[amount] if dp[amount] != \
           float('inf') else -1

print(coinChange([1,5,11], 15))  # 3 (5+5+5)
print(coinChange([2], 3))        # -1
print(coinChange([1,2,5], 11))   # 3 (5+5+1)
