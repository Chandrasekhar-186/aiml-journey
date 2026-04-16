# Day 30 — LeetCode #239 Sliding Window Maximum
# Difficulty: Hard 🔴
# Approach: Monotonic Deque
# Time: O(n) | Space: O(k)

from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()  # stores INDICES
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements from back
        # (they can never be maximum!)
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Test cases
print(maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3))
# [3,3,5,5,6,7]
print(maxSlidingWindow([1], 1))
# [1]



# Day 30 — LeetCode #438 Find All Anagrams
# Difficulty: Medium ⚡
# Approach: Sliding Window + Counter
# Time: O(n) | Space: O(1)

from collections import Counter

def findAnagrams(s, p):
    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window = Counter(s[:len(p)])
    result = []

    if window == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        # Add new character
        window[s[i]] += 1

        # Remove old character
        old = s[i - len(p)]
        window[old] -= 1
        if window[old] == 0:
            del window[old]

        if window == p_count:
            result.append(i - len(p) + 1)

    return result

print(findAnagrams("cbaebabacd", "abc"))
# [0, 6]
print(findAnagrams("abab", "ab"))
# [0, 1, 2]
