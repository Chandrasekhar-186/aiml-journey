# CodeSignal Practice 3 — Hard Patterns
# Date: May 7, 2026
# Focus: Problems that trip candidates up!

# ── HARD PATTERN 1: LRU Cache ──────────────────
# Frequently appears on CodeSignal!
# Time: O(1) per operation | Space: O(capacity)

from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))    # 1
cache.put(3, 3)        # evicts key 2
print(cache.get(2))    # -1 (evicted!)
print(cache.get(3))    # 3


# ── HARD PATTERN 2: Word Ladder ────────────────
# BFS on implicit graph — CodeSignal favorite!
# Time: O(M²×N) | Space: O(M²×N)

from collections import deque

def ladderLength(beginWord, endWord, wordList):
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, steps = queue.popleft()

        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]

                if new_word == endWord:
                    return steps + 1

                if (new_word in word_set and
                        new_word not in visited):
                    visited.add(new_word)
                    queue.append((new_word,
                                   steps + 1))
    return 0

print(ladderLength("hit","cog",
    ["hot","dot","dog","lot","log","cog"]))  # 5
print(ladderLength("hit","cog",
    ["hot","dot","dog","lot","log"]))        # 0


# ── HARD PATTERN 3: Median of Two Sorted Arrays ─
# Binary search on partition — classic Hard!
# Time: O(log(min(m,n))) | Space: O(1)

def findMedianSortedArrays(nums1, nums2):
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n) // 2

    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2  # partition nums1
        j = half - i        # partition nums2

        # Values around partition
        left1  = nums1[i-1] if i > 0 else float('-inf')
        right1 = nums1[i]   if i < m else float('inf')
        left2  = nums2[j-1] if j > 0 else float('-inf')
        right2 = nums2[j]   if j < n else float('inf')

        if left1 <= right2 and left2 <= right1:
            # Found correct partition!
            if (m + n) % 2 == 1:
                return float(min(right1, right2))
            return (max(left1, left2) +
                    min(right1, right2)) / 2.0
        elif left1 > right2:
            hi = i - 1
        else:
            lo = i + 1

print(findMedianSortedArrays([1,3],[2]))      # 2.0
print(findMedianSortedArrays([1,2],[3,4]))    # 2.5


# ── HARD PATTERN 4: Serialize/Deserialize Tree ─
# Coding interview classic — tests tree mastery!

from collections import deque

class Codec:
    def serialize(self, root):
        if not root:
            return "null"
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("null")
        return ",".join(result)

    def deserialize(self, data):
        if data == "null":
            return None
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        return root

print("LRU, Word Ladder, Median, Serialize: ✅")
print("All hard patterns covered!")
