# Day 26 — LeetCode #42 Trapping Rain Water
# Difficulty: Hard 🔴
# Approach: Two pointers
# Time: O(n) | Space: O(1)

def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water

# Test cases
print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
print(trap([4,2,0,3,2,5]))               # 9
print(trap([3,0,2,0,4]))                 # 7



# Day 26 — LeetCode #371 Sum Without + operator
# Difficulty: Medium ⚡
# Approach: Bit manipulation
# Time: O(1) | Space: O(1)

def getSum(a, b):
    mask = 0xFFFFFFFF  # 32-bit mask

    while b & mask:
        carry = (a & b) << 1
        a = a ^ b
        b = carry

    # Handle negative numbers
    return a if b == 0 else a & mask

print(getSum(1, 2))    # 3
print(getSum(-1, 1))   # 0
print(getSum(3, 5))    # 8
