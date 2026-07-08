# CodeSignal Sim 3 — Day 100 Edition
# Timer: 70 minutes, camera on simulation
# Read all 4 first, then solve easiest→hardest

print("CodeSignal Sim 3 — START TIMER NOW!")

# ─────────────────────────────
# P1: Easy (10 min target)
# Valid Parentheses with wildcards
# LeetCode #678
# ─────────────────────────────
def checkValidString(s):
    """
    '*' can be '(', ')' or ''
    Track range of possible open counts
    lo = minimum possible open parens
    hi = maximum possible open parens
    """
    lo = hi = 0
    for c in s:
        if c == '(':
            lo += 1; hi += 1
        elif c == ')':
            lo -= 1; hi -= 1
        else:  # '*'
            lo -= 1; hi += 1
        if hi < 0: return False
        lo = max(lo, 0)
    return lo == 0

print(checkValidString("()"))     # True
print(checkValidString("(*)"))    # True
print(checkValidString("(*))"))   # True
print(checkValidString("((("))    # False

# ─────────────────────────────
# P2: Easy (10 min target)
# Product of Array Except Self
# LeetCode #238
# ─────────────────────────────
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n

    # Left pass: result[i] = product of left
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]

    # Right pass: multiply by product of right
    suffix = 1
    for i in range(n-1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]

    return result

print(productExceptSelf([1,2,3,4]))  # [24,12,8,6]
print(productExceptSelf([-1,1,0,-3,3]))

# ─────────────────────────────
# P3: Medium (20 min target)
# Longest Substring with K Distinct
# LeetCode #340 (Premium → solve anyway)
# ─────────────────────────────
from collections import defaultdict

def lengthOfLongestSubstringKDistinct(s, k):
    if k == 0: return 0
    window = defaultdict(int)
    left = result = 0

    for right, char in enumerate(s):
        window[char] += 1

        while len(window) > k:
            left_char = s[left]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
            left += 1

        result = max(result, right-left+1)

    return result

print(lengthOfLongestSubstringKDistinct(
    "eceba", 2))   # 3 ("ece")
print(lengthOfLongestSubstringKDistinct(
    "aa", 1))      # 2

# ─────────────────────────────
# P4: Medium-Hard (30 min target)
# Time Based Key Value + follow-up
# (Databricks intel problem variant)
# ─────────────────────────────
class TimeBasedKV:
    """Extended TimeMap with TTL + delete"""
    def __init__(self, default_ttl=None):
        self.store = defaultdict(list)
        self.ttl = default_ttl
        self.deleted = set()

    def set(self, key, value, timestamp):
        # Don't store if key was deleted
        # after its creation
        self.store[key].append(
            (timestamp, value)
        )

    def get(self, key, timestamp):
        if key not in self.store:
            return ""
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

    def delete(self, key, timestamp):
        """Mark key as deleted at timestamp"""
        self.deleted.add((key, timestamp))

    def get_with_delete(self, key, timestamp):
        """Respect deletions"""
        # Check if deleted before timestamp
        for del_key, del_ts in self.deleted:
            if del_key == key and \
               del_ts <= timestamp:
                return ""
        return self.get(key, timestamp)

# Test
kv = TimeBasedKV()
kv.set("foo", "bar", 1)
kv.set("foo", "bar2", 4)
print(kv.get("foo", 3))   # "bar"
print(kv.get("foo", 5))   # "bar2"
kv.delete("foo", 3)
print(kv.get_with_delete("foo", 5))  # ""
print(kv.get_with_delete("foo", 2))  # "bar"

print("\nCodeSignal Sim 3: 4/4 complete! ✅")

# Score estimate:
# P1 Easy:   250 points
# P2 Easy:   250 points
# P3 Medium: 250 points
# P4 Medium: 200 points (partial ok)
# Target:    850+ / 1000
