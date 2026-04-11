# Day 27 — LeetCode #295 Find Median Stream
# Difficulty: Hard 🔴
# Approach: Two heaps (max-heap + min-heap)
# Time: O(log n) add | O(1) findMedian

import heapq

class MedianFinder:
    def __init__(self):
        # max-heap for lower half (negate for Python)
        self.lower = []
        # min-heap for upper half
        self.upper = []

    def addNum(self, num):
        # Always add to lower first
        heapq.heappush(self.lower, -num)

        # Balance: lower's max <= upper's min
        if (self.lower and self.upper and
                -self.lower[0] > self.upper[0]):
            val = -heapq.heappop(self.lower)
            heapq.heappush(self.upper, val)

        # Balance sizes (lower can have 1 extra)
        if len(self.lower) > len(self.upper) + 1:
            val = -heapq.heappop(self.lower)
            heapq.heappush(self.upper, val)
        elif len(self.upper) > len(self.lower):
            val = heapq.heappop(self.upper)
            heapq.heappush(self.lower, -val)

    def findMedian(self):
        if len(self.lower) > len(self.upper):
            return -self.lower[0]
        return (-self.lower[0] + self.upper[0]) / 2

# Test
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 1.5
mf.addNum(3)
print(mf.findMedian())  # 2.0


# Day 27 — LeetCode #739 Daily Temperatures
# Difficulty: Medium ⚡
# Approach: Monotonic Stack
# Time: O(n) | Space: O(n)

def dailyTemperatures(temperatures):
    stack = []  # stores indices
    result = [0] * len(temperatures)

    for i, temp in enumerate(temperatures):
        # Pop while current temp > stack top temp
        while (stack and
               temp > temperatures[stack[-1]]):
            idx = stack.pop()
            result[idx] = i - idx  # days waited!
        stack.append(i)

    return result

print(dailyTemperatures(
    [73,74,75,71,69,72,76,73]))
# [1,1,4,2,1,1,0,0]
print(dailyTemperatures([30,40,50,60]))
# [1,1,1,0]
