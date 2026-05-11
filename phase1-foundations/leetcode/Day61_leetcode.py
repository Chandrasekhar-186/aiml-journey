# Speed challenge — 10 min max!
def isValidBST(root):
    def validate(node, lo, hi):
        if not node: return True
        if node.val <= lo or node.val >= hi:
            return False
        return (validate(node.left, lo,
                          node.val) and
                validate(node.right,
                          node.val, hi))
    return validate(root,
                    float('-inf'),
                    float('inf'))

# Phase 3 Day 3 — LeetCode #105
# Construct Binary Tree
# Difficulty: Medium ⚡
# Approach: Recursive divide and conquer
# Time: O(n) | Space: O(n)

def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None

    # Root is always first in preorder!
    root = TreeNode(preorder[0])

    # Find root position in inorder
    mid = inorder.index(preorder[0])

    # Left subtree: mid elements in inorder
    # Right subtree: rest
    root.left = buildTree(
        preorder[1:mid+1],
        inorder[:mid]
    )
    root.right = buildTree(
        preorder[mid+1:],
        inorder[mid+1:]
    )
    return root
