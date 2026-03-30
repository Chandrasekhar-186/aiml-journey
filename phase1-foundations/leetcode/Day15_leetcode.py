# Day 15 — LeetCode #153 Find Min Rotated Array
# Difficulty: Medium ⚡
# Approach: Modified Binary Search
# Time: O(log n) | Space: O(1)

def findMin(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            # Min is in RIGHT half
            left = mid + 1
        else:
            # Min is in LEFT half (including mid)
            right = mid

    return nums[left]

# Test cases
print(findMin([3,4,5,1,2]))     # 1
print(findMin([4,5,6,7,0,1,2])) # 0
print(findMin([11,13,15,17]))   # 11

# Day 15 — LeetCode #15 3Sum
# Difficulty: Medium ⚡
# Approach: Sort + Two Pointers
# Time: O(n²) | Space: O(1)

def threeSum(nums):
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i],
                                nums[left],
                                nums[right]])
                # Skip duplicates
                while (left < right and
                       nums[left] == nums[left+1]):
                    left += 1
                while (left < right and
                       nums[right] == nums[right-1]):
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result

print(threeSum([-1,0,1,2,-1,-4]))  # [[-1,-1,2],[-1,0,1]]
print(threeSum([0,0,0]))            # [[0,0,0]]
