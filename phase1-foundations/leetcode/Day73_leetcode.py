# Speed — 2 min max! Should be instant now!
def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target: return m
        elif nums[m] < target: l = m + 1
        else: r = m - 1
    return -1

print(search([-1,0,3,5,9,12], 9))  # 4


# Phase 3 CV Day 2 — LeetCode #33
# Search in Rotated Sorted Array
# Difficulty: Medium ⚡
# Approach: Modified binary search
# Time: O(log n) | Space: O(1)

def search(nums, target):
    l, r = 0, len(nums) - 1

    while l <= r:
        m = (l + r) // 2

        if nums[m] == target:
            return m

        # Left half is sorted
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1  # target in left
            else:
                l = m + 1  # target in right
        # Right half is sorted
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1  # target in right
            else:
                r = m - 1  # target in left

    return -1

print(search([4,5,6,7,0,1,2], 0))   # 4
print(search([4,5,6,7,0,1,2], 3))   # -1
print(search([1], 0))                # -1
