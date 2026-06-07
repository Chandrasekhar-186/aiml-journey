# Speed — 3 min max!
def hammingWeight(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
# OR: return bin(n).count('1')

print(hammingWeight(11))   # 3 (1011)
print(hammingWeight(128))  # 1 (10000000)

# Phase 4 Day 4 — LeetCode #190
# Reverse Bits
# Difficulty: Easy-Medium ⚡
# Approach: Bit manipulation
# Time: O(32) = O(1) | Space: O(1)

def reverseBits(n):
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

print(reverseBits(0b00000010100101000001111010011100))
# 964176192
print(reverseBits(0b11111111111111111111111111111101))
# 3221225471

# How it works:
# Each iteration:
# 1. Shift result left (make room)
# 2. OR in last bit of n
# 3. Shift n right (consume bit)
# After 32 iterations: bits reversed!
