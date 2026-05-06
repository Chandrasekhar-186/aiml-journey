## Advanced SQL Functions — Quick Reference

array_max(col)     → max value in array
array_min(col)     → min value in array
array_size(col)    → length of array
element_at(col, n) → nth element (1-indexed!)
                     negative = from end
filter(col, fn)    → filter array elements
transform(col, fn) → map over array elements
aggregate(col, zero, merge) → reduce array
flatten(col)       → nested → flat array

## Window Functions Complete
rank()             → 1,2,2,4 (ties skip)
dense_rank()       → 1,2,2,3 (no skip)
row_number()       → 1,2,3,4 (always unique)
percent_rank()     → 0.0 to 1.0 relative rank
ntile(n)           → divide into n buckets
lag(col, n)        → value n rows before
lead(col, n)       → value n rows after
first_value(col)   → first in window
last_value(col)    → last in window

## Pivot Pattern
groupBy("row_key")
.pivot("col_to_pivot", [optional values])
.agg(F.first("value"))

## Higher-Order Functions (Spark 3.0+)
F.filter(array, lambda x: condition)
F.transform(array, lambda x: expression)
F.aggregate(array, zero, lambda acc,x: merge)
F.exists(array, lambda x: condition)
F.forall(array, lambda x: condition)

## Stack for Nested Problems
Push state when entering new scope
Pop state when leaving scope
Examples: decode string, valid parens,
          nested calculations, AST evaluation

## Phase 2 Final Stats
Days complete:  25/30
LeetCode:       107+ problems
Speed targets:  6 core patterns < 10 min each
Project:        COMPLETE + polished
Certs:          2 free earned
Mock exams:     4 (83-88% consistent)
CodeSignal:     Practice sessions done
