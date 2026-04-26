# Databricks Spark Associate Developer
# Full Mock Test — Day 41
# Date: April 21, 2026
# Target: 50+/60

## SECTION 1: Spark Architecture (12 questions)

Q1: What component converts logical plan to stages?
a) TaskScheduler  b) DAGScheduler
c) SparkContext   d) ClusterManager
Answer: b) DAGScheduler

Q2: What is a stage boundary in Spark?
a) End of a job
b) Point where a shuffle occurs
c) When executor memory is full
d) When cache() is called
Answer: b) Shuffle boundary creates new stage

Q3: Driver program responsibilities include:
a) Executing tasks in parallel
b) Storing cached RDD partitions
c) Maintaining SparkContext + scheduling
d) Managing cluster resources
Answer: c) Driver = brain, executors = muscle

Q4: What triggers a Spark job?
a) filter()  b) select()  c) groupBy()  d) count()
Answer: d) count() — it's an ACTION!

Q5: Default number of shuffle partitions?
a) 100  b) 200  c) 500  d) Dynamic
Answer: b) 200

Q6: What does lazy evaluation mean?
a) Spark is slow by default
b) Transformations build DAG, execute on action
c) Data is loaded lazily from disk
d) Executors wait for driver instructions
Answer: b) DAG built lazily, executed at action

Q7: Which is NOT a Spark transformation?
a) map()  b) filter()  c) collect()  d) join()
Answer: c) collect() is an ACTION

Q8: Spark executor memory regions (3 main):
a) Reserved, User, Unified
b) Driver, Worker, Cache
c) JVM, Python, Native
d) Heap, Stack, Off-heap
Answer: a) Reserved + User + Unified

Q9: What does mapPartitions() provide over map()?
a) Better parallelism
b) Processes partition as iterator
   (efficient for DB connections!)
c) Automatic caching
d) No shuffle
Answer: b) Iterator per partition

Q10: Stage → Task relationship:
a) 1 stage = 1 task
b) 1 stage = many tasks (1 per partition)
c) Many stages = 1 task
d) Tasks create stages
Answer: b) One task per partition per stage

Q11: What is the Catalyst optimizer?
a) Spark's cluster manager
b) Query optimization engine
   (4 phases: analysis→logical→physical→codegen)
c) Memory manager
d) Streaming engine
Answer: b) Query optimizer

Q12: Tungsten's whole-stage code generation:
a) Generates Python code
b) Generates optimized JVM bytecode
   eliminating virtual function calls
c) Compresses data
d) Manages shuffle
Answer: b) JVM bytecode generation

## SECTION 2: DataFrame API (15 questions)

Q13: How to add column with constant value?
a) df.add("col", 5)
b) df.withColumn("col", F.lit(5))
c) df.select(F.const(5))
d) df.append("col", 5)
Answer: b) F.lit() for literal values

Q14: What does df.explain(extended=True) show?
a) Schema only
b) All 4 optimizer plan phases
c) Row count
d) Partition info
Answer: b) Parsed+Analyzed+Optimized+Physical

Q15: Difference between repartition() + coalesce()?
a) Same thing
b) repartition=full shuffle (any n),
   coalesce=reduce only (no shuffle)
c) coalesce causes shuffle
d) repartition only reduces
Answer: b) Key difference: shuffle behavior

Q16: Which causes a shuffle?
a) filter()  b) select()
c) distinct()  d) withColumn()
Answer: c) distinct() → shuffle!

Q17: How to read CSV with header in Spark?
a) spark.read.csv(path)
b) spark.read.option("header","true").csv(path)
c) spark.read.csv(path, header=True)
d) Both b and c
Answer: d) Both syntaxes valid!

Q18: What does cache() use as storage level?
a) DISK_ONLY
b) MEMORY_AND_DISK
c) MEMORY_ONLY
d) OFF_HEAP
Answer: c) MEMORY_ONLY

Q19: How to rename a column?
a) df.rename("old","new")
b) df.withColumnRenamed("old","new")
c) df.select(F.col("old").alias("new"))
d) Both b and c
Answer: d) Both work!

Q20: What is predicate pushdown?
a) Moving joins to reduce side
b) Moving filters closer to data source
   (reduces data read from storage!)
c) Pushing predicates to Python
d) Sort optimization
Answer: b) Filter early = less I/O!

Q21: F.broadcast() hint does what?
a) Sends DataFrame to all workers
b) Replicates small table to avoid shuffle
c) Caches DataFrame
d) Increases parallelism
Answer: b) Broadcast join = no shuffle!

Q22: What does df.persist(MEMORY_AND_DISK) do?
a) Same as cache()
b) Stores in memory, spills to disk if full
c) Stores only on disk
d) Stores in off-heap memory
Answer: b) Memory first, disk fallback

Q23: Column pruning optimization means:
a) Removing duplicate columns
b) Dropping unused columns before reading
c) Compressing column data
d) Reordering columns
Answer: b) Only read needed columns!

Q24: What does df.coalesce(1) produce?
a) 1 partition without shuffle
b) 1 partition with shuffle
c) Splits into 1-byte partitions
d) Removes all null values
Answer: a) Reduce to 1 partition, NO shuffle

Q25: How to get top 5 by score?
a) df.head(5)
b) df.orderBy(F.desc("score")).limit(5)
c) df.top(5)
d) df.take(5)
Answer: b) orderBy + limit

Q26: Window function PARTITION BY equivalent:
a) Window.groupBy("col")
b) Window.partitionBy("col")
c) Window.splitBy("col")
d) Window.orderBy("col")
Answer: b) Window.partitionBy()

Q27: What is AQE (Adaptive Query Execution)?
a) Async query engine
b) Runtime query optimization
   (coalesces partitions, fixes skew,
    converts join strategies)
c) Automatic caching
d) Advanced queue engine
Answer: b) Runtime optimization!

## SECTION 3: Spark SQL (10 questions)

Q28: Register DataFrame as temp view:
a) df.createView("name")
b) df.createOrReplaceTempView("name")
c) spark.register(df, "name")
d) df.toSQL("name")
Answer: b) createOrReplaceTempView

Q29: RANK() vs DENSE_RANK() difference:
a) Same
b) RANK: 1,2,2,4 (skips) vs
   DENSE_RANK: 1,2,2,3 (no skip)
c) DENSE_RANK skips numbers
d) RANK is faster
Answer: b) RANK skips, DENSE_RANK doesn't

Q30: HAVING vs WHERE:
a) Same
b) WHERE: pre-aggregation filter,
   HAVING: post-aggregation filter
c) HAVING is faster
d) WHERE works only with joins
Answer: b) WHERE=row level, HAVING=group level

Q31: LAG() function purpose:
a) Adds delay to query
b) Accesses previous row's value
c) Calculates running total
d) Filters null values
Answer: b) Previous row value!

Q32: Broadcast join syntax in SQL:
a) /*+ BROADCAST(table) */
b) df.hint("broadcast")
c) Both a and b work
d) Neither — use config only
Answer: c) Both hint syntaxes valid!

Q33: What does EXPLAIN FORMATTED show?
a) Schema
b) Human-readable physical plan
c) Row count
d) Memory usage
Answer: b) Formatted physical plan

Q34: How to run SQL on DataFrame?
a) df.sql("SELECT...")
b) spark.sql("SELECT...") after createTempView
c) SparkSQL.run("SELECT...")
d) sc.sql("SELECT...")
Answer: b) spark.sql() after registering view

Q35: ROW_NUMBER() vs RANK():
a) Same
b) ROW_NUMBER: always unique (1,2,3,4)
   RANK: ties get same rank (1,2,2,4)
c) RANK always unique
d) ROW_NUMBER allows ties
Answer: b) ROW_NUMBER always unique!

Q36: Constant folding optimization:
a) Folds DataFrame columns
b) Evaluates constant expressions at plan time
   (2+3 → 5 before execution)
c) Removes constant columns
d) Folds shuffle partitions
Answer: b) Pre-compute at plan time!

Q37: How to write partitioned Delta table?
a) df.write.partition("col").save(path)
b) df.write.format("delta")
         .partitionBy("col").save(path)
c) df.write.delta(partitionBy="col")
d) df.partitionBy("col").write.delta()
Answer: b) .format("delta").partitionBy()

## SECTION 4: Delta Lake (10 questions)

Q38: Delta Lake ACID — what provides atomicity?
a) Parquet format
b) Transaction log (_delta_log)
c) Spark executor
d) Checkpointing
Answer: b) Transaction log!

Q39: How to query Delta table version 5?
a) spark.read.delta.version(5).load(path)
b) spark.read.format("delta")
         .option("versionAsOf",5).load(path)
c) DeltaTable.version(5)
d) delta.timeTravel(5)
Answer: b) versionAsOf option!

Q40: OPTIMIZE command does what?
a) Updates statistics
b) Compacts small files into larger ones
c) Removes deleted rows
d) Updates schema
Answer: b) File compaction!

Q41: VACUUM removes:
a) Duplicate rows
b) Old data files beyond retention period
c) Empty partitions
d) Null values
Answer: b) Old files > retention threshold!

Q42: MERGE operation in Delta:
a) Combines two tables permanently
b) Upsert: update existing + insert new
c) Merges schemas
d) Joins two Delta tables
Answer: b) UPSERT = update + insert!

Q43: Z-ORDER BY (col) does what?
a) Sorts data alphabetically
b) Co-locates related data in same files
   for faster filtered queries
c) Creates partitions
d) Compresses by column
Answer: b) Data co-location for fast reads!

Q44: Change Data Feed provides:
a) Schema changes only
b) Row-level changes (insert/update/delete)
c) Partition changes
d) File-level changes
Answer: b) Row-level change tracking!

Q45: Schema evolution in Delta requires:
a) Rewriting entire table
b) .option("mergeSchema","true") on write
c) Dropping and recreating table
d) Manual ALTER TABLE only
Answer: b) mergeSchema option!

Q46: Delta checkpoint files created every:
a) 5 versions  b) 10 versions
c) 50 versions d) 100 versions
Answer: b) Every 10 versions!

Q47: Liquid Clustering advantage over Z-ORDER:
a) Faster writes
b) Incremental clustering, changeable columns
c) Better compression
d) Smaller files
Answer: b) Incremental + flexible!

## SECTION 5: Streaming (8 questions)

Q48: Streaming output modes (3):
a) read, write, update
b) append, complete, update
c) batch, stream, micro
d) once, always, never
Answer: b) append + complete + update!

Q49: Watermark purpose in streaming:
a) Adds timestamps to events
b) Bounds state size by dropping old late data
c) Marks checkpoints
d) Filters invalid events
Answer: b) State management + late data!

Q50: Checkpoint location stores:
a) Processed results
b) Current offsets + state + metadata
c) Schema information
d) Partition info
Answer: b) Offsets + state + metadata!

Q51: foreachBatch provides:
a) Row-by-row processing
b) Full DataFrame API for each micro-batch
c) Automatic caching
d) Schema validation
Answer: b) DataFrame API per batch!

Q52: Trigger(once=True) means:
a) Process one message
b) Process all available data then stop
c) Run once per hour
d) Single partition processing
Answer: b) Process all then stop!

Q53: readStream differs from read by:
a) Only works with Kafka
b) Returns streaming DataFrame (isStreaming=True)
c) Requires schema
d) Slower performance
Answer: b) Returns streaming DataFrame!

Q54: Session window groups events by:
a) Fixed time intervals
b) Gaps in activity (dynamic duration)
c) User sessions from login data
d) Random grouping
Answer: b) Activity gap-based grouping!

Q55: Kafka offset "earliest" means:
a) Read only new messages
b) Read from beginning of topic
c) Read latest offset only
d) Read random offset
Answer: b) Start from beginning!

## SECTION 6: Optimization (5 questions)

Q56: Data skew fix — salting technique:
a) Adds salt to data for security
b) Appends random suffix to hot key
   to distribute across partitions
c) Removes skewed rows
d) Increases memory for skewed partitions
Answer: b) Random suffix = balanced distribution!

Q57: When does broadcast join NOT help?
a) Small lookup table
b) Both tables are large (>10MB each)
c) Inner join
d) With partitioned tables
Answer: b) Both large = broadcast fails!

Q58: reduceByKey better than groupByKey because:
a) Simpler API
b) Partial aggregation on map side = less shuffle
c) Faster sorting
d) Supports more operations
Answer: b) Map-side combine!

Q59: spark.sql.shuffle.partitions default:
a) 100  b) 200  c) Based on cores  d) 500
Answer: b) 200!

Q60: Best storage level for large DataFrames
      used twice in production:
a) MEMORY_ONLY
b) DISK_ONLY
c) MEMORY_AND_DISK
d) OFF_HEAP
Answer: c) MEMORY_AND_DISK (won't OOM!)

## SCORING
My answers: __/60
Target pass: 42/60 (70%)
My target:   50/60 (83%+)
