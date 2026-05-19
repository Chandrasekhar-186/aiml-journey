# Speed — 3 min max!
def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

# Phase 3 Day 11 — LeetCode #143
# Reorder List
# Difficulty: Medium ⚡
# Approach: Find middle + reverse + merge
# Time: O(n) | Space: O(1)

def reorderList(head):
    if not head or not head.next:
        return

    # Step 1: Find middle (slow/fast pointers)
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: Reverse second half
    prev = None
    curr = slow.next
    slow.next = None  # split list!
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    # Step 3: Merge two halves
    first, second = head, prev
    while second:
        tmp1 = first.next
        tmp2 = second.next
        first.next = second
        second.next = tmp1
        first = tmp1
        second = tmp2
