# Speed — 8 min max!
def isPalindrome(head):
    # Step 1: Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: Reverse second half
    prev = None
    curr = slow
    while curr:
        next_n = curr.next
        curr.next = prev
        prev = curr
        curr = next_n

    # Step 3: Compare both halves
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True


# Phase 3 CV Day 1 — LeetCode #148
# Sort List
# Difficulty: Medium ⚡
# Approach: Merge sort on linked list
# Time: O(n log n) | Space: O(log n)

def sortList(head):
    if not head or not head.next:
        return head

    # Find middle (split into two halves)
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None  # split!

    # Recursively sort both halves
    left = sortList(head)
    right = sortList(mid)

    # Merge sorted halves
    dummy = curr = ListNode(0)
    while left and right:
        if left.val <= right.val:
            curr.next = left
            left = left.next
        else:
            curr.next = right
            right = right.next
        curr = curr.next
    curr.next = left or right
    return dummy.next
