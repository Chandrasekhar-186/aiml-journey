# Day 29 — LeetCode #23 Merge K Sorted Lists
# Difficulty: Hard 🔴
# Approach: Min-heap
# Time: O(n log k) | Space: O(k)

import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    heap = []

    # Push first node of each list
    for i, node in enumerate(lists):
        if node:
            # (value, index, node) — index breaks ties!
            heapq.heappush(heap,
                           (node.val, i, node))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next

        if node.next:
            heapq.heappush(heap,
                (node.next.val, i, node.next))

    return dummy.next



# Day 29 — LeetCode #98 Validate BST
# Difficulty: Medium ⚡
# Approach: DFS with bounds
# Time: O(n) | Space: O(h)

def isValidBST(root):
    def validate(node, min_val, max_val):
        if not node:
            return True
        if (node.val <= min_val or
                node.val >= max_val):
            return False
        return (validate(node.left,
                          min_val, node.val) and
                validate(node.right,
                          node.val, max_val))

    return validate(root,
                    float('-inf'),
                    float('inf'))
