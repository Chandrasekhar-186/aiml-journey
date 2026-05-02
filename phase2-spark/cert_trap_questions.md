# Cert Precision Trap Questions
Date: April 29, 2026
Goal: Answer each INSTANTLY with zero hesitation

## CATEGORY 1: Subtle API Behaviors

TRAP 1: Is cache() eager or lazy?
WRONG: "cache() stores in memory immediately"
RIGHT: "cache() is LAZY — marks for caching
        but executes only when action called"

TRAP 2: What does df.schema return?
WRONG: df.schema() — with parentheses
RIGHT: df.schema — NO parentheses (attribute!)
       df.dtypes — also attribute, returns list
       df.printSchema() — method with ()

TRAP 3: What is coalesce(0)?
WRONG: "reduces to 0 partitions"
RIGHT: INVALID — minimum is 1!
       Spark throws IllegalArgumentException

TRAP 4: Can append mode work with aggregation?
WRONG: "yes, just like complete"
RIGHT: NO — append mode CANNOT use aggregations
       Use complete (all rows) or
       update (changed rows only)

TRAP 5: What does cache() return?
WRONG: "None" or "the cached data"
RIGHT: The SAME DataFrame (it's a transformation!)
       Enables method chaining:
       df.cache().count()  ← both valid!

TRAP 6: persist() with no arguments?
WRONG: "same as MEMORY_AND_DISK"
RIGHT: Same as cache() = MEMORY_ONLY!
       persist() == persist(MEMORY_ONLY)
       == cache()

TRAP 7: repartition(5) on 3-partition DF?
WRONG: "coalesce — no shuffle needed"
RIGHT: Full SHUFFLE always (repartition can
       increase OR decrease — always shuffles!)

TRAP 8: coalesce(10) on 3-partition DF?
WRONG: "increases to 10 partitions"
RIGHT: STAYS at 3! coalesce only REDUCES
       Cannot increase partition count!

## CATEGORY 2: Exact Numbers

TRAP 9: Default shuffle partitions?
ANSWER: 200 (spark.sql.shuffle.partitions)

TRAP 10: Delta checkpoint frequency?
ANSWER: Every 10 commits to _delta_log/

TRAP 11: Default VACUUM retention?
ANSWER: 7 days (168 hours)

TRAP 12: Default broadcast threshold?
ANSWER: 10MB (spark.sql.autoBroadcastJoinThreshold)

TRAP 13: Catalyst phases count?
ANSWER: 4 (Analysis→Logical→Physical→Codegen)

TRAP 14: Default parallelism for RDDs?
ANSWER: 2 × CPU cores (NOT 200!)
        200 is for SQL shuffle partitions only!

TRAP 15: Executor memory fraction default?
ANSWER: 0.6 (spark.memory.fraction)

## CATEGORY 3: Delta Lake Precision

TRAP 16: Time travel AFTER vacuum?
WRONG: "still works"
RIGHT: IMPOSSIBLE — files permanently deleted!
       Cannot time-travel past vacuum point!

TRAP 17: mergeSchema vs overwriteSchema?
mergeSchema=true: ADDS new columns
overwriteSchema=true: REPLACES entire schema
                      (use with caution!)

TRAP 18: CONVERT TO DELTA rewrites data?
WRONG: "yes, converts Parquet to Delta format"
RIGHT: NO data rewrite! Just adds _delta_log/
       Instant conversion of existing Parquet!

TRAP 19: Delta MERGE requires primary key?
WRONG: "yes, must have unique ID"
RIGHT: NO — uses any condition expression
       Can merge on composite keys or expressions

TRAP 20: OPTIMIZE + ZORDER rewrites all files?
WRONG: "only compacts — no rewrite"
RIGHT: YES — rewrites files with data co-located
       Expensive for large tables!

## CATEGORY 4: Streaming Precision

TRAP 21: Checkpoint required for streaming?
WRONG: "optional — streaming works without it"
RIGHT: OPTIONAL for correctness but REQUIRED
       for fault tolerance + exactly-once!
       Production = ALWAYS set checkpoint!

TRAP 22: trigger(once=True) processes:
WRONG: "one message"
RIGHT: ALL available data in ONE micro-batch
       then terminates the query!

TRAP 23: Watermark affects output mode?
WRONG: "watermark works with all output modes"
RIGHT: Watermark required for stateful ops
       but works with append + update modes
       NOT with complete (complete keeps all state)

TRAP 24: Static side in stream-static join?
WRONG: "refreshes when Delta table updates"
RIGHT: Loaded ONCE at query startup
       Must restart query to see new static data!

TRAP 25: foreachBatch vs foreach?
foreach:     row-by-row processing (slow!)
foreachBatch: full DataFrame per batch (fast!)
              Use foreachBatch for production!

## CATEGORY 5: Join Precision

TRAP 26: Left semi join columns returned?
WRONG: "columns from both tables"
RIGHT: LEFT table columns ONLY
       No right-side columns added!

TRAP 27: Cross join produces:
WRONG: "all matching rows"
RIGHT: CARTESIAN product — n×m rows!
       No join condition = every row × every row
       DANGEROUS on large tables!

TRAP 28: Broadcast join threshold exceeded?
WRONG: "Spark raises error"
RIGHT: Falls back to SortMergeJoin automatically!
       No error — just slower join!

TRAP 29: Can you broadcast both sides?
WRONG: "yes, broadcast both for speed"
RIGHT: Only ONE side can be broadcast!
       Typically the smaller side.

TRAP 30: Anti join vs left join + filter null?
Both find non-matching rows BUT:
Anti join: more efficient (no null columns added)
Left + filter: works but less efficient
Prefer: anti join for "not exists" pattern!

## PRACTICE: Answer these instantly

Say the answer out loud before reading:
1. cache() is: lazy/eager?
2. df.schema has: parentheses/no parentheses?
3. Default shuffle partitions: ___
4. coalesce(10) on 5-partition DF: ___ partitions
5. repartition(3) on 5-partition DF: shuffle y/n?
6. append mode + aggregation: valid/invalid?
7. Delta checkpoint every ___ commits
8. VACUUM retention default: ___ days
9. broadcast threshold default: ___MB
10. semi join returns columns from: left/right/both?

ANSWERS: lazy, no parens, 200, 5, yes,
         invalid, 10, 7, 10, left only
