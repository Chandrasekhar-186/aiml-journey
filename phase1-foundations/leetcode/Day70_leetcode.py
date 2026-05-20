# Speed — 5 min max!
def mergeTwoLists(l1, l2):
    dummy = curr = ListNode(0)
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next

# Phase 3 Day 12 — LeetCode #19
# Remove Nth Node From End
# Difficulty: Medium ⚡
# Approach: Two pointers (fast n ahead of slow)
# Time: O(n) | Space: O(1)

def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy

    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next

    # Remove the nth node
    slow.next = slow.next.next
    return dummy.next
