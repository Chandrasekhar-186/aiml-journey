# Day 23 — LeetCode #91 Decode Ways
# Difficulty: Medium ⚡
# Approach: Dynamic Programming
# Time: O(n) | Space: O(n)

def numDecodings(s):
    if not s or s[0] == '0':
        return 0

    n = len(s)
    # dp[i] = ways to decode s[:i]
    dp = [0] * (n + 1)
    dp[0] = 1  # empty string
    dp[1] = 1  # first char (not '0')

    for i in range(2, n + 1):
        # Single digit decode
        one_digit = int(s[i-1])
        if one_digit != 0:
            dp[i] += dp[i-1]

        # Two digit decode
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]

    return dp[n]

# Test cases
print(numDecodings("12"))     # 2 (AB or L)
print(numDecodings("226"))    # 3
print(numDecodings("06"))     # 0 (invalid)
print(numDecodings("11106"))  # 2



# Day 23 — LeetCode #55 Jump Game
# Difficulty: Medium ⚡
# Approach: Greedy — track max reachable
# Time: O(n) | Space: O(1)

def canJump(nums):
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False  # can't reach here!
        max_reach = max(max_reach, i + jump)

    return True

# Test cases
print(canJump([2,3,1,1,4]))  # True
print(canJump([3,2,1,0,4]))  # False
print(canJump([0]))           # True
