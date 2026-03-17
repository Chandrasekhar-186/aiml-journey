# Day 05 — LeetCode #70 Climbing Stairs
# Difficulty: Easy
# Approach: Dynamic Programming (Fibonacci pattern)
# Time: O(n) | Space: O(1)

def climbStairs(n):
    if n <= 2:
        return n
    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    return curr

# Test cases
print(climbStairs(2))   # 2
print(climbStairs(3))   # 3
print(climbStairs(5))   # 8


# Day 05 — LeetCode #88 Merge Sorted Array
# Difficulty: Easy
# Approach: Two pointers from the end
# Time: O(m+n) | Space: O(1)

def merge(nums1, m, nums2, n):
    p1, p2, p = m-1, n-1, m+n-1

    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

# Test
nums1 = [1,2,3,0,0,0]
merge(nums1, 3, [2,5,6], 3)
print(nums1)  # [1,2,2,3,5,6]
