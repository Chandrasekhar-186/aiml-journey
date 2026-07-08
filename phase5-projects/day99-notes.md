## Hard Problem Patterns

Sliding Window Maximum:
Monotonic deque (decreasing)
→ Remove from front: outside window
→ Remove from back: smaller than current
→ Front always = window maximum

Median Finder:
Two heaps: max-heap left + min-heap right
Invariant: max(left) <= min(right)
           |left| == |right| or |left|+1
Median: left top if odd, avg if even
KEY: Python heapq is min-heap → negate!

Min Window Substring:
need Counter + missing count
right expands: if need[char]>0: missing-=1
When missing==0: shrink left, track best
KEY: missing tracks REQUIRED characters!

## Databricks Problem Pool — All Mastered
TicTacToe:   row/col/diag sums ✅
TimeMap:     binary search on timestamps ✅
LazyArray:   ops list, execute at indexOf ✅
SnapshotArr: binary search on snap_ids ✅
IP Firewall: bit manipulation + CIDR ✅

## Expand Around Center
For each center (odd) and gap (even):
expand while s[left]==s[right]
count++ each valid palindrome
O(n²) time, O(1) space

## Day 99 → Day 100
99 days of preparation
Tomorrow: the milestone
81 days to application
One more day. Same standard.
