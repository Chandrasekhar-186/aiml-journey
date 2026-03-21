# Day 07 — LeetCode #226 Invert Binary Tree
# Difficulty: Easy
# Approach: Recursive DFS
# Time: O(n) | Space: O(h)

def invertTree(root):
    if not root:
        return None
    # Swap left and right
    root.left, root.right = root.right, root.left
    # Recursively invert subtrees
    invertTree(root.left)
    invertTree(root.right)
    return root

# Day 07 — LeetCode #206 Reverse Linked List
# Difficulty: Easy
# Approach: Iterative two pointers
# Time: O(n) | Space: O(1)

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    prev = None
    curr = head

    while curr:
        next_node = curr.next  # save next
        curr.next = prev       # reverse pointer
        prev = curr            # move prev forward
        curr = next_node       # move curr forward

    return prev  # prev is new head
