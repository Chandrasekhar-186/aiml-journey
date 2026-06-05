# Speed — 5 min max! Binary search on answer
def firstBadVersion(n):
    l, r = 1, n
    while l < r:
        mid = (l + r) // 2
        if isBadVersion(mid):
            r = mid      # bad in left half
        else:
            l = mid + 1  # bad in right half
    return l


# Phase 4 Day 2 — LeetCode #287
# Find Duplicate Number
# Difficulty: Medium ⚡
# Approach: Floyd's cycle detection!
# Time: O(n) | Space: O(1)

def findDuplicate(nums):
    # Treat as linked list
    # nums[i] = next pointer
    # Duplicate = cycle entry!
    slow = fast = nums[0]

    # Phase 1: detect cycle
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: find cycle entry
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow

print(findDuplicate([1,3,4,2,2]))  # 2
print(findDuplicate([3,1,3,4,2]))  # 3
print(findDuplicate([1,1]))        # 1
