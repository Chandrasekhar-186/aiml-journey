# Phase 3 Day 2 — LeetCode #152
# Maximum Product Subarray
# Difficulty: Medium ⚡
# Approach: Track both max AND min
# Time: O(n) | Space: O(1)

def maxProduct(nums):
    # Key insight: negative × negative = positive!
    # Track both max and min at each position
    global_max = nums[0]
    cur_max = cur_min = nums[0]

    for n in nums[1:]:
        # All 3 candidates for new max/min
        candidates = (n,
                       cur_max * n,
                       cur_min * n)
        cur_max = max(candidates)
        cur_min = min(candidates)
        global_max = max(global_max, cur_max)

    return global_max

print(maxProduct([2,3,-2,4]))    # 6
print(maxProduct([-2,0,-1]))     # 0
print(maxProduct([-2,3,-4]))     # 24
print(maxProduct([-3,-1,-1]))    # 3


# 2 minutes max — should be instant now!
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
print(climbStairs(5))  # 8
