# HARD BLITZ — Phase 4 Closing Day
# These patterns appear in Databricks OA!

# Problem 1: #42 Trapping Rain Water (20 min)
# Difficulty: Hard 🔴
# Approach: Two pointers O(n) O(1)

def trap(height):
    l, r = 0, len(height)-1
    left_max = right_max = water = 0

    while l < r:
        if height[l] < height[r]:
            if height[l] >= left_max:
                left_max = height[l]
            else:
                water += left_max - height[l]
            l += 1
        else:
            if height[r] >= right_max:
                right_max = height[r]
            else:
                water += right_max - height[r]
            r -= 1
    return water

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
print(trap([4,2,0,3,2,5]))               # 9

# Problem 2: #84 Largest Rectangle in Histogram
# (15 min)
# Difficulty: Hard 🔴
# Approach: Monotonic stack

def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights = heights + [0]  # sentinel

    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            max_area = max(max_area,
                            height*(i-idx))
            start = idx
        stack.append((start, h))
    return max_area

print(largestRectangleArea([2,1,5,6,2,3]))  # 10
print(largestRectangleArea([2,4]))           # 4

# Problem 3: #23 Merge K Sorted Lists (5 min)
# Difficulty: Hard (but with heap = easy!)
import heapq

def mergeKLists(lists):
    dummy = curr = type(
        'Node', (), {'val':0,'next':None}
    )()
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(
                heap, (node.val, i, node)
            )
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(
                heap,
                (node.next.val, i, node.next)
            )
    return dummy.next

print("Merge K Lists: heap pattern ✅")

# Problem 4: #128 Longest Consecutive Sequence
# (5 min speed)
def longestConsecutive(nums):
    num_set = set(nums)
    best = 0
    for n in num_set:
        if n-1 not in num_set:  # start of seq!
            cur = n
            length = 1
            while cur+1 in num_set:
                cur += 1
                length += 1
            best = max(best, length)
    return best

print(longestConsecutive(
    [100,4,200,1,3,2]))   # 4
print(longestConsecutive(
    [0,3,7,2,5,8,4,6,0,1]))  # 9


# Final CodeSignal OA simulation!
# Target: 850+ score

# Simulate 4 problems in 70 min
# Today: solve all 4 in 40 min (buffer!)

# P1 Easy (10 min): Two Sum
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target-n in seen:
            return [seen[target-n], i]
        seen[n] = i
    return []

# P2 Easy (10 min): Valid Parentheses
def isValid(s):
    stack = []
    pairs = {')':'(', ']':'[', '}':'{'}
    for c in s:
        if c in '([{':
            stack.append(c)
        elif not stack or \
             stack[-1] != pairs[c]:
            return False
        else:
            stack.pop()
    return not stack

# P3 Medium (15 min): Subarray Sum = K
from collections import defaultdict
def subarraySum(nums, k):
    count = defaultdict(int)
    count[0] = 1
    prefix = result = 0
    for n in nums:
        prefix += n
        result += count[prefix-k]
        count[prefix] += 1
    return result

# P4 Medium (15 min): Min Stack
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, val):
        self.stack.append(val)
        self.min_stack.append(
            min(val, self.min_stack[-1]
                if self.min_stack else val)
        )
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    def top(self): return self.stack[-1]
    def getMin(self): return self.min_stack[-1]

# Test all
print(twoSum([2,7,11,15], 9))   # [0,1]
print(isValid("()[]{}"))          # True
print(subarraySum([1,1,1], 2))   # 2
ms = MinStack()
ms.push(-2); ms.push(0); ms.push(-3)
print(ms.getMin())                # -3
print("CodeSignal sim: 4/4 ✅")
