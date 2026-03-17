## NumPy vs Pandas vs Spark DataFrames

NumPy:
→ Arrays and matrices, mathematical operations
→ In-memory, single machine
→ Use for: numerical computing, ML math

Pandas:
→ Tabular data with labels, data manipulation
→ In-memory, single machine (fits in RAM)
→ Use for: data cleaning, EDA, small datasets

Spark DataFrame:
→ Distributed tabular data across cluster
→ Lazy evaluation, optimized execution
→ Use for: big data (GBs to PBs), production

## Dynamic Programming — Plain English
Break problem into smaller subproblems.
Store results to avoid recomputing.
Climbing Stairs: f(n) = f(n-1) + f(n-2)
→ Same as Fibonacci!

## Spark SQL vs DataFrame API
Both compile to same execution plan.
SQL: better for complex joins + aggregations
API: better for programmatic/dynamic queries
