# Day 14 — LeetCode #572 Subtree of Another Tree
# Difficulty: Easy
# Approach: DFS + same tree check
# Time: O(m*n) | Space: O(h)

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        if not root:
            return False
        
        if self.isSameTree(root, subRoot):
            return True
        
        return (self.isSubtree(root.left, subRoot) or 
                self.isSubtree(root.right, subRoot))

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        if not p and not q:
            return True
        
        if not p or not q or p.val != q.val:
            return False
        
        
        return (self.isSameTree(p.left, q.left) and 
                self.isSameTree(p.right, q.right))

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
