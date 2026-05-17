# Speed — 5 min max!
def invertTree(root):
    if not root: return None
    root.left, root.right = \
        invertTree(root.right), \
        invertTree(root.left)
    return root

# Phase 3 Day 9 — LeetCode #236
# LCA of Binary Tree (not BST!)
# Difficulty: Medium ⚡
# Approach: Post-order DFS
# Time: O(n) | Space: O(h)

def lowestCommonAncestor(root, p, q):
    # Base cases
    if not root: return None
    if root == p or root == q:
        return root  # found one!

    # Search both subtrees
    left = lowestCommonAncestor(
        root.left, p, q
    )
    right = lowestCommonAncestor(
        root.right, p, q
    )

    # Both found → current node is LCA!
    if left and right:
        return root

    # One found → propagate up
    return left if left else right
