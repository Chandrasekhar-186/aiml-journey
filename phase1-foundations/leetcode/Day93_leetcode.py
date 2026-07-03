# Phase 4 Day 9 — LeetCode #55
# Jump Game
# Difficulty: Medium ⚡
# Approach: Greedy — track max reachable
# Time: O(n) | Space: O(1)

def canJump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False  # can't reach i!
        max_reach = max(max_reach, i + jump)
    return True

print(canJump([2,3,1,1,4]))  # True
print(canJump([3,2,1,0,4]))  # False
print(canJump([0]))           # True


# Phase 4 Day 9 — LeetCode #45
# Jump Game II — minimum jumps
# Difficulty: Medium ⚡
# Approach: Greedy BFS levels
# Time: O(n) | Space: O(1)

def jump(nums):
    jumps = curr_end = curr_far = 0
    for i in range(len(nums)-1):
        curr_far = max(curr_far, i+nums[i])
        if i == curr_end:
            jumps += 1
            curr_end = curr_far
    return jumps

print(jump([2,3,1,1,4]))    # 2
print(jump([2,3,0,1,4]))    # 2
print(jump([1,2,3]))         # 2
