# Phase 3 Day 8 — LeetCode #543
# Diameter of Binary Tree
# Difficulty: Easy ✅
# Approach: DFS — track max at each node
# Time: O(n) | Space: O(h)

def diameterOfBinaryTree(root):
    diameter = [0]

    def depth(node):
        if not node: return 0
        left = depth(node.left)
        right = depth(node.right)
        # Update diameter at this node
        diameter[0] = max(diameter[0],
                           left + right)
        return 1 + max(left, right)

    depth(root)
    return diameter[0]


# Phase 3 Day 8 — LeetCode #572
# Subtree of Another Tree
# Difficulty: Easy ✅
# Approach: DFS + tree equality check
# Time: O(m*n) | Space: O(h)

def isSubtree(root, subRoot):
    def isSame(s, t):
        if not s and not t: return True
        if not s or not t: return False
        return (s.val == t.val and
                isSame(s.left, t.left) and
                isSame(s.right, t.right))

    if not root: return False
    if isSame(root, subRoot): return True
    return (isSubtree(root.left, subRoot) or
            isSubtree(root.right, subRoot))
