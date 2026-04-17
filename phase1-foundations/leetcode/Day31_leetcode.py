# Phase 2 Day 1 — LeetCode #560
# Difficulty: Medium ⚡
# Approach: Prefix sum + HashMap
# Time: O(n) | Space: O(n)

from collections import defaultdict

def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    seen = defaultdict(int)
    seen[0] = 1  # empty subarray

    for num in nums:
        prefix_sum += num
        # If prefix_sum - k exists before
        # → subarray between those points = k
        count += seen[prefix_sum - k]
        seen[prefix_sum] += 1

    return count

print(subarraySum([1,1,1], 2))      # 2
print(subarraySum([1,2,3], 3))      # 2
print(subarraySum([-1,-1,1], 0))    # 1


# Phase 2 Day 1 — LeetCode #49 Group Anagrams
# Difficulty: Medium ⚡
# Approach: HashMap with sorted string key
# Time: O(n * k log k) | Space: O(n)

from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # canonical form
        groups[key].append(s)
    return list(groups.values())

print(groupAnagrams(
    ["eat","tea","tan","ate","nat","bat"]))
# [["eat","tea","ate"],["tan","nat"],["bat"]]
