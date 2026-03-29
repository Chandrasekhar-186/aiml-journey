# Day 14 — LeetCode #572 Subtree of Another Tree
# Difficulty: Easy
# Approach: DFS + same tree check
# Time: O(m*n) | Space: O(h)

def isSubtree(root, subRoot):
    if not root:
        return False
    if isSameTree(root, subRoot):
        return True
    return (isSubtree(root.left, subRoot) or
            isSubtree(root.right, subRoot))

def isSameTree(p, q):
    if not p and not q: return True
    if not p or not q: return False
    return (p.val == q.val and
            isSameTree(p.left, q.left) and
            isSameTree(p.right, q.right))

# Day 14 — LeetCode #11 Container With Most Water
# Difficulty: Medium ⚡
# Approach: Two pointers
# Time: O(n) | Space: O(1)

def maxArea(height):
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        water = min(height[left],
                    height[right]) * (right - left)
        max_water = max(max_water, water)

        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water

print(maxArea([1,8,6,2,5,4,8,3,7]))  # 49
print(maxArea([1,1]))                  # 1
