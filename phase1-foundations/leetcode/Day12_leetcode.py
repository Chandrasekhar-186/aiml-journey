# Day 12 — LeetCode #322 Coin Change
# Difficulty: Medium ⚡
# Approach: Bottom-up Dynamic Programming
# Time: O(amount * coins) | Space: O(amount)

def coinChange(coins, amount):
    # dp[i] = min coins needed for amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # base case: 0 coins for amount 0

    for amt in range(1, amount + 1):
        for coin in coins:
            if coin <= amt:
                # Use this coin + best for remainder
                dp[amt] = min(dp[amt],
                               dp[amt - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Test cases
print(coinChange([1,5,10,25], 41))  # 4 (25+10+5+1)
print(coinChange([1,2,5], 11))      # 3 (5+5+1)
print(coinChange([2], 3))           # -1 (impossible)


# Day 12 — LeetCode #238 Product Except Self
# Difficulty: Medium ⚡
# Approach: Prefix + Suffix products
# Time: O(n) | Space: O(1) output excluded

def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n

    # Left pass: result[i] = product of all left
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]

    # Right pass: multiply product of all right
    suffix = 1
    for i in range(n-1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]

    return result

print(productExceptSelf([1,2,3,4]))  # [24,12,8,6]
print(productExceptSelf([-1,1,0,-3,3]))  # [0,0,9,0,0]
