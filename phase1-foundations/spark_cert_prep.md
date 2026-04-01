# Spark Cert Mock — Day 17
Date: March 29, 2026

Q1: What is lazy evaluation in Spark?
A: Transformations build a DAG but don't execute
   until an action is called.

Q2: Difference between cache() and persist()?
A: cache() = persist(MEMORY_ONLY).
   persist() lets you choose storage level.

Q3: What is a shuffle in Spark?
A: Data redistribution across partitions —
   triggered by groupBy, join, distinct.
   Most expensive operation in Spark!

Q4: When to use broadcast join?
A: When one DataFrame is small (<10MB).
   Sends small DF to all workers —
   avoids expensive shuffle.

Q5: What is a Spark partition?
A: Chunk of data processed by one task.
   More partitions = more parallelism.
   Default: 200 after shuffle.

Q6: Difference between DataFrame and RDD?
A: DataFrame = structured, optimized by Catalyst.
   RDD = unstructured, manual optimization.
   Always prefer DataFrame API!

Q7: What does explain() show?
A: Physical execution plan — how Spark
   will actually execute your query.

Q8: What is Delta Lake ACID?
A: Atomicity, Consistency, Isolation, Durability.
   Prevents corrupt data in concurrent writes.

Q9: What is Z-Ordering in Delta Lake?
A: Co-locates related data in same files.
   Dramatically speeds up queries with filters.

Q10: What is the Catalyst optimizer?
A: Spark's query optimizer — automatically
   rewrites queries for best performance.
   Works on DataFrame/SQL API only (not RDD!).
