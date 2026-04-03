# Day 19 — LeetCode #215 Kth Largest Element
# Difficulty: Medium ⚡
# Approach: Min-heap of size k
# Time: O(n log k) | Space: O(k)

import heapq

def findKthLargest(nums, k):
    # Min-heap of size k
    # Smallest element in heap = kth largest overall
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest

    return heap[0]  # kth largest!

# Test cases
print(findKthLargest([3,2,1,5,6,4], 2))   # 5
print(findKthLargest([3,2,3,1,2,4,5,5,6], 4))  # 4

# Alternative: QuickSelect O(n) average
import random
def findKthLargest_quick(nums, k):
    k = len(nums) - k  # kth largest = (n-k)th smallest

    def quickselect(l, r):
        pivot = nums[r]
        p = l
        for i in range(l, r):
            if nums[i] <= pivot:
                nums[i], nums[p] = nums[p], nums[i]
                p += 1
        nums[p], nums[r] = nums[r], nums[p]
        if p > k: return quickselect(l, p-1)
        if p < k: return quickselect(p+1, r)
        return nums[p]

    return quickselect(0, len(nums)-1)


# Day 19 — LeetCode #347 Top K Frequent Elements
# Difficulty: Medium ⚡
# Approach: Heap + Counter
# Time: O(n log k) | Space: O(n)

from collections import Counter
import heapq

def topKFrequent(nums, k):
    count = Counter(nums)
    # heapq.nlargest uses heap internally
    return heapq.nlargest(
        k, count.keys(),
        key=count.get
    )

# Alternative: bucket sort O(n)
def topKFrequent_bucket(nums, k):
    count = Counter(nums)
    # Bucket index = frequency
    freq = [[] for _ in range(len(nums)+1)]
    for num, cnt in count.items():
        freq[cnt].append(num)

    result = []
    for i in range(len(freq)-1, 0, -1):
        result.extend(freq[i])
        if len(result) >= k:
            return result[:k]

print(topKFrequent([1,1,1,2,2,3], 2))  # [1,2]
print(topKFrequent([1], 1))             # [1]
