# Speed — 8 min max!
def findMin(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        m = (l + r) // 2
        if nums[m] > nums[r]:
            l = m + 1  # min in right half
        else:
            r = m      # min in left half (or m)
    return nums[l]

print(findMin([3,4,5,1,2]))   # 1
print(findMin([4,5,6,7,0,1,2]))  # 0

# Phase 3 CV Day 3 — LeetCode #74
# Search a 2D Matrix
# Difficulty: Medium ⚡
# Approach: Binary search on flattened matrix
# Time: O(log(m*n)) | Space: O(1)

def searchMatrix(matrix, target):
    m, n = len(matrix), len(matrix[0])
    l, r = 0, m * n - 1

    while l <= r:
        mid = (l + r) // 2
        # Convert 1D index to 2D!
        row, col = divmod(mid, n)
        val = matrix[row][col]

        if val == target:
            return True
        elif val < target:
            l = mid + 1
        else:
            r = mid - 1

    return False

print(searchMatrix(
    [[1,3,5,7],[10,11,16,20],[23,30,34,60]],
    3))   # True
print(searchMatrix(
    [[1,3,5,7],[10,11,16,20],[23,30,34,60]],
    13))  # False
