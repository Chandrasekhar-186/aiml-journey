# Day 20 — LeetCode #300 LIS
# Difficulty: Medium ⚡ (Hard approach)
# Approach 1: DP O(n²)
# Approach 2: Binary Search O(n log n)

def lengthOfLIS(nums):
    # Approach 1: Standard DP
    n = len(nums)
    dp = [1] * n  # each element is LIS of length 1

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Approach 2: O(n log n) with binary search
import bisect
def lengthOfLIS_fast(nums):
    tails = []  # tails[i] = smallest tail of IS length i+1

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)

print(lengthOfLIS([10,9,2,5,3,7,101,18]))    # 4
print(lengthOfLIS_fast([10,9,2,5,3,7,101,18])) # 4
print(lengthOfLIS([0,1,0,3,2,3]))              # 4


# Day 20 — LeetCode #56 Merge Intervals
# Difficulty: Medium ⚡
# Approach: Sort + greedy merge
# Time: O(n log n) | Space: O(n)

def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            # Overlapping — extend current interval
            merged[-1][1] = max(merged[-1][1], end)
        else:
            # Non-overlapping — add new interval
            merged.append([start, end])

    return merged

print(merge([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6],[8,10],[15,18]]
print(merge([[1,4],[4,5]]))
# [[1,5]]
