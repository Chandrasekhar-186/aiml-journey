# Phase 3 LLM Day 4 — LeetCode #424
# Longest Repeating Char Replacement
# Difficulty: Medium ⚡
# Approach: Sliding window
# Time: O(n) | Space: O(1)

def characterReplacement(s, k):
    count = {}
    max_count = left = result = 0

    for right in range(len(s)):
        count[s[right]] = \
            count.get(s[right], 0) + 1
        max_count = max(max_count,
                         count[s[right]])

        # Window invalid: need > k replacements
        while (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1

        result = max(result,
                      right - left + 1)

    return result

print(characterReplacement("ABAB", 2))   # 4
print(characterReplacement("AABABBA", 1))# 4


# Speed revisit — 10 min max!
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()  # stores indices
    result = []
    for i, num in enumerate(nums):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

print(maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3))
# [3,3,5,5,6,7]
