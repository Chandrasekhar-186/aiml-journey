#Two Sum — LeetCode #1
#Problem
#Given an array of integers and a target, return indices of two numbers that add up to target.
#Brute Force — O(n²)
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
