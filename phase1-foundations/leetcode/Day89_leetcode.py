# Phase 4 Day 5 — LeetCode #371
# Sum of Two Integers (no + or - !)
# Difficulty: Medium ⚡
# Approach: Bit manipulation XOR + AND
# Time: O(1) | Space: O(1)

def getSum(a, b):
    # Python has infinite precision ints
    # Need 32-bit mask
    MASK = 0xFFFFFFFF
    MAX = 0x7FFFFFFF

    while b & MASK:
        # XOR = sum without carry
        # AND << 1 = carry
        carry = (a & b) << 1
        a = (a ^ b) & MASK
        b = carry & MASK

    # Handle negative numbers
    return a if a <= MAX else ~(a ^ MASK)

print(getSum(1, 2))    # 3
print(getSum(-2, 3))   # 1
print(getSum(-1, -1))  # -2

# Speed — 2 min max! Classic XOR trick
from functools import reduce
from operator import xor

def singleNumber(nums):
    return reduce(xor, nums)
    # OR: return reduce(lambda a,b: a^b, nums)

print(singleNumber([2,2,1]))      # 1
print(singleNumber([4,1,2,1,2]))  # 4
