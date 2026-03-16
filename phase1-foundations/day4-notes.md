## Window Functions — Plain English
A window function performs calculation across a 
set of rows RELATED to current row, without 
collapsing them into one row (unlike GROUP BY)

GROUP BY: 5 rows → 2 rows (collapsed)
WINDOW:   5 rows → 5 rows (keeps all, adds column)

## Key window functions:
ROW_NUMBER → unique sequential number (no ties)
RANK       → same rank for ties, skips next number
DENSE_RANK → same rank for ties, no skip
LAG        → value from previous row
LEAD       → value from next row

## Spark lazy evaluation:
Transformations (filter, select, groupBy) = LAZY
→ builds a DAG, nothing runs yet

Actions (show, count, collect) = EAGER  
→ triggers actual computation

This is why Spark is fast — it optimizes the
entire DAG before running anything!
```
