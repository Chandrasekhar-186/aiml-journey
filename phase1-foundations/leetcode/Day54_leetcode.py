# Phase 2 Day 24 — LeetCode #1094 Car Pooling
# Difficulty: Medium ⚡
# Approach: Difference array (sweep line)
# Time: O(n log n) | Space: O(n)

def carPooling(trips, capacity):
    # Difference array approach
    changes = {}
    for passengers, start, end in trips:
        changes[start] = changes.get(start, 0) \
                         + passengers
        changes[end] = changes.get(end, 0) \
                       - passengers

    current = 0
    for pos in sorted(changes.keys()):
        current += changes[pos]
        if current > capacity:
            return False
    return True

print(carPooling([[2,1,5],[3,3,7]], 4))   # False
print(carPooling([[2,1,5],[3,3,7]], 5))   # True
print(carPooling([[3,2,7],[3,7,9],[8,3,9]], 11))  # True


# Speed challenge — 10 min max!
from collections import defaultdict

def subarraySum(nums, k):
    count = defaultdict(int)
    count[0] = 1
    prefix = result = 0
    for n in nums:
        prefix += n
        result += count[prefix - k]
        count[prefix] += 1
    return result

print(subarraySum([1,1,1], 2))  # 2
print(subarraySum([1,2,3], 3))  # 2
