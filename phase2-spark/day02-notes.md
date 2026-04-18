## Catalyst — 4 Phases (memorize for cert!)
1. Analysis:          resolve names + types
2. Logical Opt:       pushdown, pruning, folding
3. Physical Planning: join strategy selection
4. Code Generation:   Tungsten bytecode

## Key Optimizations Catalyst Does FOR YOU
Predicate Pushdown: filter moved before join
Column Pruning:     unused cols dropped early
Constant Folding:   1+1 evaluated at plan time
Join Reordering:    smaller table as build side

## Tungsten — 3 Innovations
1. Explicit memory management (no JVM GC!)
2. Cache-aware data layout (CPU friendly)
3. Whole-stage code generation (no virtual calls)

## AQE — 3 Runtime Optimizations
1. Coalesce small partitions automatically
2. Convert sort-merge → broadcast join
3. Fix data skew by splitting large partitions

## CRITICAL: Never use Python UDFs!
Python UDF breaks Tungsten code generation
→ Data crosses Python/JVM boundary
→ No vectorization possible
→ Can be 10-100× slower!
Use: F.when(), F.expr(), built-in functions
If must use UDF → use Pandas UDF (vectorized!)

## Phase 2 LeetCode → Spark Connection
Top K Frequent = groupBy().count().limit(k)
Frequency Sort  = Window + orderBy(freq)
Subarray Sum    = Running total = Spark cumsum
Group Anagrams  = groupBy(sorted_key).collect()
