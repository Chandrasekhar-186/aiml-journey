# Speed challenge — 8 min max!LeetCode #235
def lowestCommonAncestor(root, p, q):
    while root:
        if p.val < root.val and \
           q.val < root.val:
            root = root.left
        elif p.val > root.val and \
             q.val > root.val:
            root = root.right
        else:
            return root  # split point = LCA!

# Phase 3 Day 4 — LeetCode #230
# Kth Smallest in BST
# Difficulty: Medium ⚡
# Approach: Inorder traversal (sorted order!)
# Time: O(k) | Space: O(h)

def kthSmallest(root, k):
    # Inorder traversal = sorted order for BST!
    stack = []
    curr = root
    count = 0

    while stack or curr:
        # Go left as far as possible
        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        count += 1
        if count == k:
            return curr.val

        curr = curr.right  # try right subtree

# Recursive approach (simpler):
def kthSmallest_recursive(root, k):
    result = []
    def inorder(node):
        if not node or len(result) == k:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)
    inorder(root)
    return result[k-1]
