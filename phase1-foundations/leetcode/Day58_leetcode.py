# Phase 2 Day 28 — LeetCode #295
# Find Median from Data Stream
# Difficulty: Hard 🔴
# Approach: Two heaps (max-heap + min-heap)
# Time: O(log n) add | O(1) find
# Space: O(n)

import heapq

class MedianFinder:
    def __init__(self):
        # Max-heap for lower half (negate!)
        self.lower = []
        # Min-heap for upper half
        self.upper = []

    def addNum(self, num):
        # Add to lower half first
        heapq.heappush(self.lower, -num)

        # Ensure lower max <= upper min
        if (self.lower and self.upper and
                -self.lower[0] > self.upper[0]):
            val = -heapq.heappop(self.lower)
            heapq.heappush(self.upper, val)

        # Balance sizes (lower can be +1)
        if len(self.lower) > len(self.upper) + 1:
            val = -heapq.heappop(self.lower)
            heapq.heappush(self.upper, val)
        elif len(self.upper) > len(self.lower):
            val = heapq.heappop(self.upper)
            heapq.heappush(self.lower, -val)

    def findMedian(self):
        if len(self.lower) > len(self.upper):
            return float(-self.lower[0])
        return (-self.lower[0] +
                 self.upper[0]) / 2.0

mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())   # 1.5
mf.addNum(3)
print(mf.findMedian())   # 2.0
mf.addNum(7)
print(mf.findMedian())   # 2.5


# Speed challenge — 10 min max!
import heapq

def mergeKLists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap,
                (node.val, i, node))
    dummy = cur = ListNode(0)
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap,
                (node.next.val, i, node.next))
    return dummy.next
