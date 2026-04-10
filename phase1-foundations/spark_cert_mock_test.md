# Spark Cert Full Mock Test — Day 26
Date: April 7, 2026
Target score: 27+/30 (90%+)
Real exam: 42/60 to pass (70%)

## SECTION 1: DataFrame API (10 questions)

Q1: Which operation causes a shuffle?
a) filter()  b) select()  c) groupBy()  d) withColumn()
Answer: c) groupBy()

Q2: What does coalesce(1) do?
a) Creates 1 partition with shuffle
b) Creates 1 partition WITHOUT shuffle
c) Splits into equal partitions
d) Caches the DataFrame
Answer: b) Without shuffle (unlike repartition!)

Q3: How do you add a column with constant value?
a) df.add("col", 5)
b) df.withColumn("col", F.lit(5))
c) df.select("col", 5)
d) df.append("col", F.const(5))
Answer: b) F.lit() for literal values

Q4: What is the output of df.count()?
a) A DataFrame  b) A Long number
c) A list        d) An RDD
Answer: b) Action returns Long (triggers execution!)

Q5: How do you rename a column?
a) df.rename("old", "new")
b) df.withColumnRenamed("old", "new")
c) df.select(F.col("old").name("new"))
d) df.alias("old", "new")
Answer: b) withColumnRenamed()

Q6: What does cache() do in Spark?
a) Saves to disk permanently
b) Stores in memory for reuse
c) Creates a checkpoint
d) Writes to Delta Lake
Answer: b) MEMORY_ONLY storage level

Q7: Which is a transformation (not action)?
a) show()   b) count()   c) filter()   d) collect()
Answer: c) filter() — lazy, returns DataFrame

Q8: How do you read CSV with header?
a) spark.read.csv("path")
b) spark.read.option("header","true").csv("path")
c) spark.read.csv("path", header=True)
d) Both b and c
Answer: d) Both b and c work!

Q9: What does explain() show?
a) Schema of DataFrame
b) Physical execution plan
c) Row count
d) Column statistics
Answer: b) Physical + logical execution plans

Q10: How to get distinct values?
a) df.unique()  b) df.distinct()
c) df.dedupe()  d) df.dropDups()
Answer: b) df.distinct() — causes shuffle!

## SECTION 2: Spark SQL (8 questions)

Q11: How to register a temp view?
a) df.createView("name")
b) df.createOrReplaceTempView("name")
c) spark.registerView(df, "name")
d) df.toSQL("name")
Answer: b) createOrReplaceTempView()

Q12: What does RANK() vs DENSE_RANK() do?
a) Both same
b) RANK skips numbers after ties, DENSE_RANK doesn't
c) DENSE_RANK skips, RANK doesn't
d) RANK is faster
Answer: b) RANK: 1,2,2,4 | DENSE_RANK: 1,2,2,3

Q13: What is LAG() used for?
a) Delay execution
b) Access previous row's value
c) Sort descending
d) Filter nulls
Answer: b) LAG(col, offset) = value N rows before

Q14: HAVING vs WHERE — key difference?
a) No difference
b) WHERE filters before GROUP BY,
   HAVING filters after GROUP BY
c) HAVING is faster
d) WHERE works only with joins
Answer: b) WHERE=pre-aggregation, HAVING=post!

Q15: How to write SQL in PySpark?
a) spark.query("SELECT...")
b) spark.sql("SELECT...")
c) df.sql("SELECT...")
d) pyspark.sql("SELECT...")
Answer: b) spark.sql()

Q16: What is a broadcast join hint?
a) /*+ BROADCAST(table) */
b) spark.broadcast(df)
c) df.hint("broadcast")
d) Both a and c
Answer: d) Both syntaxes work!

Q17: Window function PARTITION BY equivalent in Python?
a) Window.partitionBy("col")
b) Window.groupBy("col")
c) Window.splitBy("col")
d) Window.divideBy("col")
Answer: a) Window.partitionBy()

Q18: How to unpivot/melt in Spark?
a) df.unpivot()
b) df.melt()
c) df.stack()
d) Use stack() SQL function or manual union
Answer: d) No direct unpivot — use SQL stack()

## SECTION 3: Optimization (7 questions)

Q19: What is AQE in Spark 3.0+?
a) Automatic Query Execution
b) Adaptive Query Execution —
   optimizes at runtime
c) Advanced Query Engine
d) Async Query Evaluation
Answer: b) AQE — dynamically optimizes plan!

Q20: When does Spark create a new stage?
a) Every transformation
b) Every action
c) When there's a shuffle boundary
d) Every 100 operations
Answer: c) Shuffle = stage boundary!

Q21: What is data skew?
a) Corrupted data
b) Uneven partition sizes causing slow tasks
c) Schema mismatch
d) Memory overflow
Answer: b) Some partitions much larger → slow!
Fix: salting, repartition by skewed key

Q22: Best way to handle small file problem?
a) Increase executor memory
b) OPTIMIZE command in Delta Lake
c) Add more partitions
d) Use cache()
Answer: b) Delta OPTIMIZE compacts small files!

Q23: What is predicate pushdown?
a) Pushing filters closer to data source
b) Pushing joins to workers
c) Moving predicates to WHERE clause
d) Optimizing GROUP BY
Answer: a) Filter data BEFORE loading → less I/O!

Q24: Difference between map() and mapPartitions()?
a) No difference
b) map() operates per row,
   mapPartitions() per partition
c) mapPartitions() is slower
d) map() needs RDD
Answer: b) mapPartitions() more efficient for
           per-partition operations (DB connections!)

Q25: What is the default shuffle partition count?
a) 100  b) 200  c) 500  d) Dynamic
Answer: b) 200 — set spark.sql.shuffle.partitions

## SECTION 4: Delta Lake (5 questions)

Q26: What does VACUUM do in Delta Lake?
a) Removes duplicate rows
b) Deletes old version files beyond retention
c) Compacts small files
d) Updates schema
Answer: b) Removes files older than retention period
           Default: 7 days

Q27: How to enable schema evolution in Delta?
a) option("mergeSchema", "true")
b) delta.enableSchemaEvolution = true
c) ALTER TABLE ADD COLUMN
d) Both a and c
Answer: d) Both work for different scenarios!

Q28: What is Z-Ordering?
a) Sorting by Z column
b) Clustering data by column values
   for faster filtered queries
c) Partitioning strategy
d) Compression algorithm
Answer: b) Co-locates related data for faster
           point queries and range filters!

Q29: How to query Delta table version 3?
a) spark.read.delta.version(3).load(path)
b) spark.read.format("delta")
         .option("versionAsOf", 3).load(path)
c) DeltaTable.version(3)
d) delta.timeTravel(3)
Answer: b) versionAsOf option!

Q30: What makes Delta Lake ACID compliant?
a) Encryption
b) Transaction log (_delta_log)
   records every change atomically
c) Parquet format
d) Spark integration
Answer: b) _delta_log = write-ahead log
           ensures atomicity + consistency!

## MY SCORE: __/30
## Target:   27+/30 ✅
