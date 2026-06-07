# Speed — 3 min max!
def missingNumber(nums):
    n = len(nums)
    return n*(n+1)//2 - sum(nums)

print(missingNumber([3,0,1]))    # 2
print(missingNumber([0,1]))      # 2
print(missingNumber([9,6,4,2,3,5,7,0,1]))  # 8


# Phase 4 Day 3 — LeetCode #338
# Counting Bits
# Difficulty: Easy-Medium ⚡
# Approach: DP with bit pattern!
# Time: O(n) | Space: O(n)

def countBits(n):
    dp = [0] * (n + 1)
    # dp[i] = dp[i >> 1] + (i & 1)
    # Right shift removes last bit
    # & 1 checks if last bit is 1
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp

print(countBits(2))   # [0,1,1]
print(countBits(5))   # [0,1,1,2,1,2]

# Pattern: dp[i] = bits(i//2) + last_bit
# Even numbers: same bits as i//2
# Odd numbers:  same bits as i//2 + 1
