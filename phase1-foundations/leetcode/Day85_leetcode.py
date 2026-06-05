# Speed — 5 min max!
def twoSum(numbers, target):
    l, r = 0, len(numbers) - 1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target: return [l+1, r+1]
        elif s < target: l += 1
        else: r -= 1

print(twoSum([2,7,11,15], 9))   # [1,2]
print(twoSum([2,3,4], 6))       # [1,3]

# Phase 4 Day 1 — LeetCode #162
# Find Peak Element
# Difficulty: Medium ⚡
# Approach: Binary search O(log n)!
# Key: always move toward higher neighbor

def findPeakElement(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > nums[mid+1]:
            # Peak is in left half (or mid)
            r = mid
        else:
            # Peak is in right half
            l = mid + 1
    return l

print(findPeakElement([1,2,3,1]))      # 2
print(findPeakElement([1,2,1,3,5,6,4])) # 5
print(findPeakElement([1]))             # 0
