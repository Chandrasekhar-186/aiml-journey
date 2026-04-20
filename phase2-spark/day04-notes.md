## RDD Transformations (lazy)
map()            → one-to-one transformation
flatMap()        → one-to-many (split/explode)
filter()         → keep matching elements
mapPartitions()  → function per partition
                   (better for DB connections!)
reduceByKey()    → combine values per key
                   (map-side combine → less shuffle!)
groupByKey()     → group values per key
                   (NO map-side combine → SLOW!)
sortByKey()      → sort by key
join()           → inner join two RDDs
leftOuterJoin()  → left join two RDDs

## RDD Actions (trigger execution)
collect()  → return all to driver (careful!)
count()    → count rows
first()    → first element
take(n)    → first n elements
top(n)     → top n elements
reduce()   → aggregate with function
fold()     → reduce with zero value
aggregate()→ most flexible aggregation
             (different seqOp + combOp!)

## reduceByKey vs groupByKey
reduceByKey: partial combine on MAP SIDE first
             → much less data shuffled!
             → ALWAYS prefer this!
groupByKey:  shuffle ALL values then group
             → can cause OOM on large data

## Data Skew — Signs + Solutions
Signs:
→ One task 10× slower than others
→ Spark UI shows uneven partition sizes
→ OOM on specific tasks

Solutions:
1. Salting: add random suffix to hot key
   → splits hot partition into N parts
   → requires 2-phase aggregation
2. AQE: auto-detects + fixes at runtime
   → Enable spark.sql.adaptive.skewJoin.enabled
3. Repartition by different key
4. Broadcast join if possible

## 2D DP Pattern
dp[i][j] = answer using first i of A, j of B
If match: dp[i][j] = dp[i-1][j-1] + 1
Else:     dp[i][j] = max(dp[i-1][j], dp[i][j-1])
Applications: LCS, Edit Distance, Coin Change 2D
