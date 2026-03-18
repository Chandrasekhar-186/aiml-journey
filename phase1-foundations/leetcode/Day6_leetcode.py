# Day 06 — LeetCode #100 Same Tree
# Difficulty: Easy
# Approach: Recursive DFS
# Time: O(n) | Space: O(h)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSameTree(p, q):
    # Both null → same
    if not p and not q:
        return True
    # One null, one not → different
    if not p or not q:
        return False
    # Values differ → different
    if p.val != q.val:
        return False
    # Recursively check left and right
    return (isSameTree(p.left, q.left) and
            isSameTree(p.right, q.right))


# Day 06 — LeetCode #104 Max Depth Binary Tree
# Difficulty: Easy
# Approach: Recursive DFS
# Time: O(n) | Space: O(h)

def maxDepth(root):
    if not root:
        return 0
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    return 1 + max(left_depth, right_depth)
