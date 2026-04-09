# Day 25 — LeetCode #128 Longest Consecutive
# Difficulty: Medium ⚡
# Approach: HashSet — O(n) time!
# Time: O(n) | Space: O(n)

def longestConsecutive(nums):
    num_set = set(nums)
    best = 0

    for n in num_set:
        # Only start counting from sequence start!
        if n - 1 not in num_set:
            curr = n
            streak = 1

            while curr + 1 in num_set:
                curr += 1
                streak += 1

            best = max(best, streak)

    return best

print(longestConsecutive(
    [100,4,200,1,3,2]))   # 4 (1,2,3,4)
print(longestConsecutive(
    [0,3,7,2,5,8,4,6,0,1]))  # 9


# Day 25 — LeetCode #416 Partition Equal Subset
# Difficulty: Medium ⚡
# Approach: 0/1 Knapsack DP
# Time: O(n * target) | Space: O(target)

def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    dp = {0}  # achievable sums

    for num in nums:
        dp = dp | {s + num for s in dp
                   if s + num <= target}

    return target in dp

print(canPartition([1,5,11,5]))   # True (1+5+5=11)
print(canPartition([1,2,3,5]))    # False
print(canPartition([3,3,3,4,5]))  # True
