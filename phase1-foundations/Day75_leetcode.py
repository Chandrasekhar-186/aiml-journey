# Phase 3 CV Day 4 — LeetCode #875
# Koko Eating Bananas
# Difficulty: Medium ⚡
# Approach: Binary search on answer!
# Time: O(n log m) | Space: O(1)

import math

def minEatingSpeed(piles, h):
    # Binary search on speed k
    l, r = 1, max(piles)

    while l < r:
        k = (l + r) // 2
        # Hours needed at speed k
        hours = sum(math.ceil(p/k)
                    for p in piles)
        if hours <= h:
            r = k   # can go slower!
        else:
            l = k + 1  # need faster!
    return l

print(minEatingSpeed([3,6,7,11], 8))   # 4
print(minEatingSpeed([30,11,23,4,20],
                      5))               # 30


# Speed challenge — same pattern as #875!
def shipWithinDays(weights, days):
    l = max(weights)   # min capacity
    r = sum(weights)   # max capacity

    while l < r:
        mid = (l + r) // 2
        need = curr = 1
        for w in weights:
            if curr + w > mid:
                need += 1
                curr = 0
            curr += w
        if need <= days:
            r = mid
        else:
            l = mid + 1
    return l

print(shipWithinDays(
    [1,2,3,4,5,6,7,8,9,10], 5))  # 15
