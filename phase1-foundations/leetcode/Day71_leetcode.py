# Speed — 4 min max! Floyd's algorithm
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


# Phase 3 Day 13 — LeetCode #142
# Find Cycle Start
# Difficulty: Medium ⚡
# Approach: Floyd's + math proof
# Time: O(n) | Space: O(1)

def detectCycle(head):
    slow = fast = head

    # Phase 1: detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # no cycle!

    # Phase 2: find cycle start
    # Mathematical proof:
    # Let F = distance to cycle start
    # Let C = cycle length
    # When they meet: fast = slow + k*C
    # Moving slow from head AND
    # moving meeting point at same speed:
    # both reach cycle start simultaneously!
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow  # cycle start!

print("Floyd's cycle detection:")
print("Phase 1: detect with slow/fast")
print("Phase 2: reset slow to head")
print("Both advance by 1 → meet at cycle start!")
