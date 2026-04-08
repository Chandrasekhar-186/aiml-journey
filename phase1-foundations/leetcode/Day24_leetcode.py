# Day 24 — LeetCode #152 Max Product Subarray
# Difficulty: Medium ⚡ (tricky!)
# Approach: Track both max AND min products
# Time: O(n) | Space: O(1)

def maxProduct(nums):
    result = max(nums)
    cur_min = cur_max = 1

    for n in nums:
        if n == 0:
            cur_min = cur_max = 1
            continue
        # Save cur_max before overwriting!
        temp = cur_max * n
        cur_max = max(n, cur_max*n, cur_min*n)
        cur_min = min(n, temp, cur_min*n)
        result = max(result, cur_max)

    return result

# Test cases
print(maxProduct([2,3,-2,4]))    # 6
print(maxProduct([-2,0,-1]))     # 0
print(maxProduct([-2,3,-4]))     # 24
print(maxProduct([-2,-3,-4]))    # 12


# Day 24 — LeetCode #435 Non-overlapping Intervals
# Difficulty: Medium ⚡
# Approach: Greedy — sort by end time
# Time: O(n log n) | Space: O(1)

def eraseOverlapIntervals(intervals):
    if not intervals:
        return 0

    # Sort by END time — greedy key insight!
    intervals.sort(key=lambda x: x[1])
    removals = 0
    prev_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start < prev_end:
            # Overlap! Remove current interval
            removals += 1
            # Keep the one with earlier end
        else:
            prev_end = end

    return removals

print(eraseOverlapIntervals(
    [[1,2],[2,3],[3,4],[1,3]]))  # 1
print(eraseOverlapIntervals(
    [[1,2],[1,2],[1,2]]))         # 2
print(eraseOverlapIntervals(
    [[1,2],[2,3]]))               # 0
