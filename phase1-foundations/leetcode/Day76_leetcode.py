# Databricks 2026 Intel Problems
# These appeared in ACTUAL interviews!

# ════════════════════════════════
# PROBLEM 1: Tic Tac Toe (#348)
# ════════════════════════════════
# Key insight: track row/col/diagonal sums!
# +1 for player 1, -1 for player 2
# Win = |sum| == n

class TicTacToe:
    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = 0
        self.anti = 0

    def move(self, row: int, col: int,
              player: int) -> int:
        n = self.n
        v = 1 if player == 1 else -1

        self.rows[row] += v
        self.cols[col] += v
        if row == col:
            self.diag += v
        if row + col == n - 1:
            self.anti += v

        if (abs(self.rows[row]) == n or
            abs(self.cols[col]) == n or
            abs(self.diag) == n or
            abs(self.anti) == n):
            return player
        return 0

# Test
ttt = TicTacToe(3)
print(ttt.move(0,0,1))  # 0
print(ttt.move(0,2,2))  # 0
print(ttt.move(2,2,1))  # 0
print(ttt.move(1,1,2))  # 0
print(ttt.move(2,0,1))  # 0
print(ttt.move(1,0,2))  # 0
print(ttt.move(2,1,1))  # 1 ← player 1 wins!


# ════════════════════════════════
# PROBLEM 2: Time Based KV (#981)
# ════════════════════════════════
# Binary search on sorted timestamps!

from collections import defaultdict

class TimeMap:
    def __init__(self):
        # {key: [(ts, val), ...]}
        self.store = defaultdict(list)

    def set(self, key: str, value: str,
             timestamp: int) -> None:
        # timestamps always increasing!
        self.store[key].append(
            (timestamp, value)
        )

    def get(self, key: str,
             timestamp: int) -> str:
        if key not in self.store:
            return ""
        pairs = self.store[key]
        # Binary search: largest ts <= timestamp
        lo, hi = 0, len(pairs) - 1
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            if pairs[mid][0] <= timestamp:
                result = pairs[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

# Test
tm = TimeMap()
tm.set("foo", "bar", 1)
print(tm.get("foo", 1))   # "bar"
print(tm.get("foo", 3))   # "bar"
tm.set("foo", "bar2", 4)
print(tm.get("foo", 4))   # "bar2"
print(tm.get("foo", 5))   # "bar2"
print(tm.get("foo", 0))   # ""


# ════════════════════════════════
# PROBLEM 3: Snapshot Array (#1146)
# ════════════════════════════════
# Store only CHANGES, not full copies!

class SnapshotArray:
    def __init__(self, length: int):
        # Each index: list of (snap_id, val)
        self.data = [
            [[0, 0]] for _ in range(length)
        ]
        self.snap_id = 0

    def set(self, index: int,
             val: int) -> None:
        # If same snap_id → overwrite
        if (self.data[index][-1][0] ==
                self.snap_id):
            self.data[index][-1][1] = val
        else:
            self.data[index].append(
                [self.snap_id, val]
            )

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int,
             snap_id: int) -> int:
        arr = self.data[index]
        # Binary search for snap_id
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if arr[mid][0] <= snap_id:
                lo = mid
            else:
                hi = mid - 1
        return arr[lo][1]

# Test
sa = SnapshotArray(3)
sa.set(0, 5)
print(sa.snap())        # 0
sa.set(0, 6)
print(sa.get(0, 0))     # 5
print(sa.get(0, 1))     # 6 (current)


# ════════════════════════════════
# PROBLEM 4: Incremental Encoding
# (Databricks Coding Round!)
# ════════════════════════════════

def incremental_encode(arr):
    """
    In-place: arr[i] = arr[i] - arr[i-1]
    arr[0] unchanged
    Traverse BACKWARD to avoid overwrite!
    """
    for i in range(len(arr)-1, 0, -1):
        arr[i] = arr[i] - arr[i-1]
    return arr

print(incremental_encode([1, 3, 6, 10]))
# [1, 2, 3, 4]
print(incremental_encode([5, 5, 5, 5]))
# [5, 0, 0, 0]

def incremental_decode(arr):
    """Reverse: prefix sum!"""
    for i in range(1, len(arr)):
        arr[i] = arr[i] + arr[i-1]
    return arr

print(incremental_decode([1, 2, 3, 4]))
# [1, 3, 6, 10] ← original restored!


# ════════════════════════════════
# PROBLEM 5: Lazy Array
# (Databricks Coding Round!)
# ════════════════════════════════

class LazyArray:
    def __init__(self, data):
        self.data = data
        self.ops = []  # pending operations!

    def map(self, fn):
        """Lazy — don't execute yet!"""
        new = LazyArray(self.data)
        new.ops = self.ops + [fn]
        return new  # return NEW LazyArray!

    def indexOf(self, target):
        """Execute all ops, find index"""
        for i, val in enumerate(self.data):
            result = val
            for op in self.ops:
                result = op(result)
            if result == target:
                return i
        return -1

# Test
arr = LazyArray([10, 20, 30, 40, 50])
print(arr.map(lambda x: x*2).indexOf(40))
# 1 (20*2=40 at index 1)

print(arr.map(lambda x: x*2)
         .map(lambda x: x*3)
         .indexOf(240))
# 3 (40*2=80, 80*3=240 at index 3)

print(arr.map(lambda x: x+5).indexOf(25))
# 1 (20+5=25 at index 1)
