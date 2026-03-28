# Day 13 — LeetCode #198 House Robber
# Difficulty: Medium ⚡
# Approach: Dynamic Programming
# Time: O(n) | Space: O(1)

def rob(nums):
    if len(nums) == 1:
        return nums[0]

    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1,           # skip house i
                   prev2 + nums[i]) # rob house i
        prev2 = prev1
        prev1 = curr

    return prev1

# Test cases
print(rob([1,2,3,1]))      # 4 (rob 1+3)
print(rob([2,7,9,3,1]))    # 12 (rob 2+9+1)
print(rob([2,1]))           # 2

# Day 13 — LeetCode #3 Longest Substring
# Difficulty: Medium ⚡
# Approach: Sliding Window + HashSet
# Time: O(n) | Space: O(min(m,n))

def lengthOfLongestSubstring(s):
    char_set = set()
    left = 0
    max_len = 0

    for right in range(len(s)):
        # Shrink window until no duplicate
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len

# Test cases
print(lengthOfLongestSubstring("abcabcbb")) # 3
print(lengthOfLongestSubstring("bbbbb"))    # 1
print(lengthOfLongestSubstring("pwwkew"))   # 3
