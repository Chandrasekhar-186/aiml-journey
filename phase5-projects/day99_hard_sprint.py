# Day 99 — LeetCode Hard Sprint
# Date: July 6, 2026
# Maximum sharpness before Day 100!

print("="*60)
print("Day 99 — Hard Sprint + Speed Drills")
print("="*60)

# ════════════════════════════════════════
# HARD PROBLEM 1: #239 Sliding Window Max
# Time: 20 min | Target: 15 min
# ════════════════════════════════════════
from collections import deque

def maxSlidingWindow(nums, k):
    """
    Monotonic deque — always decreasing!
    Front = max of current window
    """
    dq = deque()  # stores INDICES
    result = []

    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove indices of smaller elements
        # (they'll never be the max!)
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Window fully formed
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

print("=== Sliding Window Max ===")
print(maxSlidingWindow(
    [1,3,-1,-3,5,3,6,7], 3))  # [3,3,5,5,6,7]
print(maxSlidingWindow([1], 1))    # [1]
print(maxSlidingWindow([1,-1], 1)) # [1,-1]

# ════════════════════════════════════════
# HARD PROBLEM 2: #295 Median from Stream
# Time: 20 min | Target: 18 min
# ════════════════════════════════════════
import heapq

class MedianFinder:
    """
    Two heaps: max-heap (left) + min-heap (right)
    Left half (max-heap): store negated for Python!
    Right half (min-heap): standard

    Invariant:
    - len(left) == len(right) OR
    - len(left) == len(right) + 1
    - max(left) <= min(right)
    """
    def __init__(self):
        self.left = []   # max-heap (negated!)
        self.right = []  # min-heap

    def addNum(self, num: int) -> None:
        # Always push to left first
        heapq.heappush(self.left, -num)

        # Balance: ensure max(left) <= min(right)
        if (self.right and
                -self.left[0] > self.right[0]):
            val = -heapq.heappop(self.left)
            heapq.heappush(self.right, val)

        # Size balance: left can have at most 1 more
        if len(self.left) > len(self.right) + 1:
            val = -heapq.heappop(self.left)
            heapq.heappush(self.right, val)
        elif len(self.right) > len(self.left):
            val = heapq.heappop(self.right)
            heapq.heappush(self.left, -val)

    def findMedian(self) -> float:
        if len(self.left) == len(self.right):
            return (-self.left[0] +
                     self.right[0]) / 2.0
        return float(-self.left[0])

print("\n=== Median Finder ===")
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 1.5
mf.addNum(3)
print(mf.findMedian())  # 2.0
mf.addNum(7)
mf.addNum(5)
mf.addNum(4)
print(mf.findMedian())  # 3.5

# ════════════════════════════════════════
# HARD PROBLEM 3: #76 Min Window Substr
# Time: 15 min SPEED | Already solved!
# ════════════════════════════════════════
from collections import Counter

def minWindow(s, t):
    """Speed drill — should be fast now!"""
    if not t or not s: return ""
    need = Counter(t)
    missing = len(t)
    best_l = best_r = 0
    best_len = float('inf')
    left = 0

    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if right - left < best_len:
                best_len = right - left
                best_l, best_r = left, right
            need[s[left]] += 1
            missing += 1
            left += 1

    return s[best_l:best_r] \
           if best_len < float('inf') else ""

print("\n=== Min Window Substring ===")
print(minWindow("ADOBECODEBANC","ABC"))  # "BANC"
print(minWindow("a","a"))               # "a"

# ════════════════════════════════════════
# DATABRICKS INTEL SPEED DRILLS
# Target: all in < 3 min each!
# ════════════════════════════════════════
print("\n=== Databricks Intel Speed Drills ===")

# Drill 1: TicTacToe (O(1) win detection)
class TicTacToe:
    def __init__(self, n):
        self.n = n
        self.rows = [0]*n
        self.cols = [0]*n
        self.diag = self.anti = 0

    def move(self, row, col, player):
        v = 1 if player == 1 else -1
        self.rows[row] += v
        self.cols[col] += v
        if row == col: self.diag += v
        if row+col == self.n-1: self.anti += v
        n = self.n
        if (abs(self.rows[row])==n or
            abs(self.cols[col])==n or
            abs(self.diag)==n or
            abs(self.anti)==n):
            return player
        return 0

t = TicTacToe(3)
assert t.move(0,0,1)==0
assert t.move(1,1,2)==0
assert t.move(2,2,1)==0
assert t.move(0,2,2)==0
assert t.move(0,1,1)==0
assert t.move(1,0,2)==0
assert t.move(2,1,1)==1  # player 1 wins!
print("TicTacToe: ✅ (< 2 min)")

# Drill 2: TimeMap (#981)
from collections import defaultdict
import bisect

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key, value, timestamp):
        self.store[key].append(
            (timestamp, value)
        )

    def get(self, key, timestamp):
        if key not in self.store: return ""
        pairs = self.store[key]
        lo, hi = 0, len(pairs)-1
        result = ""
        while lo <= hi:
            mid = (lo+hi)//2
            if pairs[mid][0] <= timestamp:
                result = pairs[mid][1]
                lo = mid+1
            else:
                hi = mid-1
        return result

tm = TimeMap()
tm.set("foo","bar",1)
assert tm.get("foo",1) == "bar"
assert tm.get("foo",3) == "bar"
tm.set("foo","bar2",4)
assert tm.get("foo",4) == "bar2"
assert tm.get("foo",0) == ""
print("TimeMap:    ✅ (< 2 min)")

# Drill 3: LazyArray
class LazyArray:
    def __init__(self, data):
        self.data = data
        self.ops = []

    def map(self, fn):
        new = LazyArray(self.data)
        new.ops = self.ops + [fn]
        return new

    def indexOf(self, target):
        for i, val in enumerate(self.data):
            result = val
            for op in self.ops:
                result = op(result)
            if result == target:
                return i
        return -1

arr = LazyArray([10,20,30,40,50])
assert arr.map(lambda x: x*2)\
           .indexOf(40) == 1
assert arr.map(lambda x: x*2)\
           .map(lambda x: x*3)\
           .indexOf(240) == 3
print("LazyArray:  ✅ (< 2 min)")

# Drill 4: SnapshotArray (#1146)
class SnapshotArray:
    def __init__(self, length):
        self.data = [[[0,0]]
                      for _ in range(length)]
        self.snap_id = 0

    def set(self, index, val):
        if self.data[index][-1][0]==self.snap_id:
            self.data[index][-1][1] = val
        else:
            self.data[index].append(
                [self.snap_id, val]
            )

    def snap(self):
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index, snap_id):
        arr = self.data[index]
        lo, hi = 0, len(arr)-1
        while lo < hi:
            mid = (lo+hi+1)//2
            if arr[mid][0] <= snap_id:
                lo = mid
            else:
                hi = mid-1
        return arr[lo][1]

sa = SnapshotArray(3)
sa.set(0,5)
sid = sa.snap()
sa.set(0,6)
assert sa.get(0, sid) == 5
assert sa.get(0, sid+1) == 6
print("SnapshotArr:✅ (< 2 min)")

# Drill 5: IP Firewall
class IpFirewall:
    def __init__(self, rules):
        self.rules = []
        for action, cidr in rules:
            ip_str, *prefix = cidr.split('/')
            prefix = int(prefix[0]) \
                     if prefix else 32
            network = self._to_int(ip_str)
            mask = self._make_mask(prefix)
            network &= mask
            self.rules.append(
                (action, network, mask)
            )

    def _to_int(self, ip):
        parts = ip.split('.')
        result = 0
        for p in parts:
            result = (result<<8) | int(p)
        return result

    def _make_mask(self, prefix):
        if prefix == 0: return 0
        return ((1<<32)-1) - \
               ((1<<(32-prefix))-1)

    def query(self, ip):
        ip_int = self._to_int(ip)
        for action, network, mask in self.rules:
            if ip_int & mask == network:
                return action
        return "DENY"

fw = IpFirewall([
    ("ALLOW","192.168.1.0/24"),
    ("DENY", "10.0.0.0/8"),
    ("ALLOW","1.2.3.4")
])
assert fw.query("192.168.1.100") == "ALLOW"
assert fw.query("10.5.6.7") == "DENY"
assert fw.query("1.2.3.4") == "ALLOW"
assert fw.query("8.8.8.8") == "DENY"
print("IP Firewall:✅ (< 3 min)")

print("\n✅ ALL DRILLS PASSED!")
print("Databricks problem pool: MASTERED! 🎯")

# ════════════════════════════════════════
# FINAL SPEED TEST: Classic patterns
# All should be < 2 min each now!
# ════════════════════════════════════════
print("\n=== Classic Pattern Speed Test ===")

# Binary search
def binary_search(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        m = (l+r)//2
        if nums[m] == target: return m
        elif nums[m] < target: l = m+1
        else: r = m-1
    return -1

# Two pointers
def two_sum_sorted(nums, target):
    l, r = 0, len(nums)-1
    while l < r:
        s = nums[l]+nums[r]
        if s == target: return [l+1,r+1]
        elif s < target: l += 1
        else: r -= 1

# Sliding window
def max_subarray(nums):
    cur = res = nums[0]
    for n in nums[1:]:
        cur = max(n, cur+n)
        res = max(res, cur)
    return res

# Fast DP
def climb_stairs(n):
    a = b = 1
    for _ in range(n-1): a, b = b, a+b
    return b

assert binary_search([1,3,5,7,9], 5) == 2
assert two_sum_sorted([2,7,11,15], 9) == [1,2]
assert max_subarray([-2,1,-3,4,-1,2,1]) == 6
assert climb_stairs(5) == 8
print("All classic patterns: ✅ INSTANT!")

print("\n" + "="*60)
print("Day 99 Hard Sprint — COMPLETE! 🏆")
print("TOMORROW: DAY 100 MILESTONE! 🎯")
print("="*60)
