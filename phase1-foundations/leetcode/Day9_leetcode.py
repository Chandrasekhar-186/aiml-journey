# Day 09 — LeetCode #704 Binary Search
# Difficulty: Easy
# Approach: Classic Binary Search
# Time: O(log n) | Space: O(1)

def search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2  # avoid overflow!

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1    # search right half
        else:
            right = mid - 1   # search left half

    return -1  # not found

# Test cases
print(search([-1,0,3,5,9,12], 9))   # 4
print(search([-1,0,3,5,9,12], 2))   # -1


# Day 09 — LeetCode #278 First Bad Version
# Difficulty: Easy
# Approach: Binary Search on answer space
# Time: O(log n) | Space: O(1)

def firstBadVersion(n):
    left, right = 1, n

    while left < right:
        mid = left + (right - left) // 2
        if isBadVersion(mid):
            right = mid      # bad version is here or earlier
        else:
            left = mid + 1   # bad version is later

    return left  # left == right == first bad version
