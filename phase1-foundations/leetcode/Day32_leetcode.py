# Phase 2 Day 2 — LeetCode #692 Top K Frequent Words
# Difficulty: Medium ⚡
# Approach: Heap with custom comparator
# Time: O(n log k) | Space: O(n)

import heapq
from collections import Counter

def topKFrequent(words, k):
    count = Counter(words)

    # Min-heap with negative freq + word
    # (so we can use min-heap as max-heap)
    # Tuple: (-freq, word) for lexicographic order
    heap = [(-freq, word)
            for word, freq in count.items()]
    heapq.heapify(heap)  # O(n)

    return [heapq.heappop(heap)[1]
            for _ in range(k)]

print(topKFrequent(
    ["i","love","leetcode","i",
     "love","coding"], 2))        # ["i","love"]
print(topKFrequent(
    ["the","day","is","sunny","the",
     "the","the","sunny","is","is"], 4))
# ["the","is","sunny","day"]



# Phase 2 Day 2 — LeetCode #451
# Difficulty: Medium ⚡
# Approach: Counter + bucket sort
# Time: O(n) | Space: O(n)

from collections import Counter

def frequencySort(s):
    count = Counter(s)

    # Bucket sort by frequency
    buckets = [''] * (len(s) + 1)
    for char, freq in count.items():
        buckets[freq] += char * freq

    return ''.join(reversed(buckets))

print(frequencySort("tree"))    # "eetr" or "eert"
print(frequencySort("cccaaa"))  # "cccaaa" or "aaaccc"
print(frequencySort("Aabb"))    # "bbAa" or "bbaA"
