## Spark Cert — Most Tricky Distinctions

repartition vs coalesce:
→ repartition: can increase OR decrease (SHUFFLE!)
→ coalesce: can only DECREASE (no shuffle)

cache() vs persist():
→ cache() = persist(MEMORY_ONLY)
→ persist() = choose your storage level

200 default shuffle partitions:
→ spark.sql.shuffle.partitions = 200
→ Change for your cluster size!

AQE (Adaptive Query Execution):
→ Spark 3.0+ feature
→ Dynamically changes plan at runtime
→ Handles skew, changes join strategies
→ Enabled by default in Spark 3.2+

## Two Pointer — Hard Problems
Trapping Rain Water key insight:
water[i] = min(max_left, max_right) - height[i]
Two pointer: always process SMALLER side
→ Because smaller side is the constraint!

## Bit Manipulation Basics
XOR (^):  1^1=0, 0^0=0, 1^0=1 (sum without carry)
AND (&):  1&1=1, else 0 (detect carry positions)
Shift (<<): multiply by 2 (move carry left)
a+b = (a^b) + ((a&b)<<1) → repeat until no carry

## GitHub Portfolio Checklist
✅ Descriptive README with badges
✅ Clear folder structure
✅ Daily commits (green squares)
✅ Repo topics/tags for discoverability
✅ Pinned repos on profile
✅ LinkedIn link in profile
