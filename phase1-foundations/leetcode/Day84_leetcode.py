# Phase 3 Final — LeetCode #15
# 3Sum
# Difficulty: Medium ⚡
# Approach: Sort + two pointers
# Time: O(n²) | Space: O(1)

def threeSum(nums):
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for i
        if i > 0 and nums[i] == nums[i-1]:
            continue

        l, r = i+1, len(nums)-1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            if total == 0:
                result.append(
                    [nums[i], nums[l], nums[r]]
                )
                # Skip duplicates
                while l < r and \
                      nums[l] == nums[l+1]:
                    l += 1
                while l < r and \
                      nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
            elif total < 0:
                l += 1
            else:
                r -= 1

    return result

print(threeSum([-1,0,1,2,-1,-4]))
# [[-1,-1,2],[-1,0,1]]
print(threeSum([0,1,1]))  # []
print(threeSum([0,0,0]))  # [[0,0,0]]



# Phase 3 Final — LeetCode #18
# 4Sum
# Difficulty: Medium ⚡
# Approach: Sort + two outer loops + two pointers
# Time: O(n³) | Space: O(1)

def fourSum(nums, target):
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n-3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1, n-2):
            if j > i+1 and \
               nums[j] == nums[j-1]:
                continue
            l, r = j+1, n-1
            while l < r:
                total = (nums[i] + nums[j] +
                          nums[l] + nums[r])
                if total == target:
                    result.append([
                        nums[i], nums[j],
                        nums[l], nums[r]
                    ])
                    while l < r and \
                          nums[l] == nums[l+1]:
                        l += 1
                    while l < r and \
                          nums[r] == nums[r-1]:
                        r -= 1
                    l += 1; r -= 1
                elif total < target:
                    l += 1
                else:
                    r -= 1
    return result

print(fourSum([1,0,-1,0,-2,2], 0))
# [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
