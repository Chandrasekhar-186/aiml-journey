# Speed — 5 min max! Prefix sum HashMap
from collections import defaultdict
def subarraySum(nums, k):
    count = defaultdict(int)
    count[0] = 1
    prefix = result = 0
    for n in nums:
        prefix += n
        result += count[prefix-k]
        count[prefix] += 1
    return result

print(subarraySum([1,1,1], 2))    # 2
print(subarraySum([1,2,3], 3))    # 2


# Phase 5 Day 4 — LeetCode #647
# Palindromic Substrings
# Difficulty: Medium ⚡
# Approach: Expand around center
# Time: O(n²) | Space: O(1)

def countSubstrings(s):
    count = 0
    n = len(s)

    def expand(left, right):
        nonlocal count
        while (left >= 0 and
               right < n and
               s[left] == s[right]):
            count += 1
            left -= 1
            right += 1

    for i in range(n):
        expand(i, i)    # odd length
        expand(i, i+1)  # even length

    return count

print(countSubstrings("abc"))   # 3
print(countSubstrings("aaa"))   # 6
print(countSubstrings("abba"))  # 6
