# Speed — 2 min max!
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)
print(isAnagram("anagram", "nagaram"))  # True
print(isAnagram("rat", "car"))          # False

# Speed — 5 min max!
from collections import defaultdict
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())

print(groupAnagrams(
    ["eat","tea","tan","ate","nat","bat"]))



# Phase 3 LLM Day 1 — LeetCode #347
# Top K Frequent Elements
# Difficulty: Medium ⚡
# Approach: Bucket sort O(n)!
# Time: O(n) | Space: O(n)

def topKFrequent(nums, k):
    count = {}
    for n in nums:
        count[n] = count.get(n, 0) + 1

    # Bucket: index = frequency
    freq = [[] for _ in range(len(nums)+1)]
    for num, cnt in count.items():
        freq[cnt].append(num)

    result = []
    for i in range(len(freq)-1, 0, -1):
        for num in freq[i]:
            result.append(num)
            if len(result) == k:
                return result

print(topKFrequent([1,1,1,2,2,3], 2))  # [1,2]
print(topKFrequent([1], 1))             # [1]
