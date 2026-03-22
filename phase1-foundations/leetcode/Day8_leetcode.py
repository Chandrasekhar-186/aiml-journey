# Day 08 — LeetCode #141 Linked List Cycle
# Difficulty: Easy
# Approach: Floyd's Tortoise and Hare
# Time: O(n) | Space: O(1)

def hasCycle(head):
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next          # moves 1 step
        fast = fast.next.next     # moves 2 steps
        if slow == fast:          # they met = cycle!
            return True

    return False  # fast reached end = no cycle

# Day 08 — LeetCode #21 Merge Two Sorted Lists
# Difficulty: Easy
# Approach: Iterative with dummy node
# Time: O(m+n) | Space: O(1)

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)    # dummy head
        curr = dummy

        while list1 and list2:
            if list1.val <= list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next

        # Attach remaining nodes
        curr.next = list1 if list1 else list2
        return dummy.next
