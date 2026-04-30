# Spark Cert Final Prep — Day 46
Date: April 26, 2026
Exam: May 11, 2026 (15 days!)
Mock score: 83%+ ✅
Target exam score: 90%+

## GUARANTEED EXAM TOPICS — Study These Cold

### 1. Transformation vs Action — MUST KNOW
Transformations (lazy, return DataFrame/RDD):
filter(), select(), withColumn(), groupBy(),
join(), union(), distinct(), orderBy(),
repartition(), coalesce(), cache(), persist(),
map(), flatMap(), mapPartitions(),
withColumnRenamed(), drop(), limit()

Actions (trigger execution, return result):
show(), count(), collect(), take(n), first(),
head(), foreach(), foreachBatch(),
write/save(), reduce(), aggregate(), fold(),
min(), max(), sum(), mean(), std()

TRICK: cache() and persist() are transformations!
They mark the RDD/DF but don't execute yet!

### 2. Spark SQL Functions — Complete List
Aggregate: count, sum, avg, min, max,
           stddev, variance, percentile_approx,
           collect_list, collect_set, first, last

Window:    rank, dense_rank, row_number,
           lag, lead, ntile,
           sum/avg/count OVER window,
           first_value, last_value

String:    upper, lower, trim, ltrim, rtrim,
           substring, concat, concat_ws,
           split, regexp_extract, regexp_replace,
           length, lpad, rpad, reverse,
           instr, locate, translate

Date/Time: current_date, current_timestamp,
           date_add, date_sub, datediff,
           date_format, year, month, dayofmonth,
           hour, minute, second,
           unix_timestamp, from_unixtime,
           to_date, to_timestamp

Null:      isNull, isNotNull, fillna,
           dropna, coalesce, nvl,
           nullif, ifnull

Array:     array, array_contains, array_distinct,
           array_intersect, array_union,
           array_except, size, array_size,
           explode, explode_outer,
           posexplode, flatten,
           sort_array, reverse, element_at,
           slice, array_min, array_max

Map:       map, map_keys, map_values,
           map_contains_key, map_from_arrays,
           explode (works on maps too!)

Misc:      lit, col, expr, when/otherwise,
           cast, iif, monotonically_increasing_id,
           spark_partition_id, input_file_name,
           struct, to_json, from_json,
           to_csv, from_csv, schema_of_json

### 3. Read/Write Options — Cert Favorites
CSV:
.option("header", "true/false")
.option("inferSchema", "true/false")
.option("delimiter", ",")
.option("nullValue", "NA")
.option("mode", "PERMISSIVE/DROPMALFORMED/FAILFAST")
.option("quote", "\"")
.option("escape", "\\")

JSON:
.option("multiLine", "true")
.option("allowSingleQuotes", "true")
.option("mode", "PERMISSIVE")

Parquet/Delta:
.option("mergeSchema", "true")
.option("versionAsOf", N)
.option("timestampAsOf", "datetime")

### 4. DataFrame Methods — Often Confused
df.schema          → StructType (attribute, not method!)
df.dtypes           → list of (name, type) tuples
df.columns          → list of column names
df.printSchema()    → prints tree format
df.describe()       → summary stats (mean,std,min,max)
df.summary()        → like describe + quartiles
df.show(n, truncate) → display n rows
df.display()        → Databricks-specific (richer UI)

df.select("col")     → keeps column as DataFrame
df.select(df.col)    → same
df["col"]           → Column object
df.col              → Column object (attribute access)

### 5. Join Behavior — Tricky Questions
Inner join:    only matching rows
Left outer:    all left + matching right (nulls for no match)
Right outer:   all right + matching left
Full outer:    all rows from both (nulls for no match)
Cross join:    cartesian (n×m rows — DANGEROUS!)
Left semi:     left rows where match EXISTS (no right cols!)
Left anti:     left rows where NO match (no right cols!)

Broadcast join: automatic if table < threshold
                OR explicit: F.broadcast(df)
Sort-merge join: default for large tables
                 requires shuffle + sort
Hash join:      for medium tables,
                no sort required

### 6. Performance Config — Know the Defaults
spark.sql.shuffle.partitions     = 200
spark.sql.autoBroadcastJoinThreshold = 10MB
spark.executor.memory            = 1g
spark.driver.memory              = 1g
spark.memory.fraction            = 0.6
spark.memory.storageFraction     = 0.5
spark.sql.adaptive.enabled       = true (3.2+)
spark.default.parallelism        = 2 × cores

### 7. Delta Lake — Complete Feature List
ACID transactions via _delta_log/
Time travel: versionAsOf, timestampAsOf
Schema evolution: mergeSchema=true
OPTIMIZE: file compaction
ZORDER BY: data co-location
VACUUM: removes old files (default 7 days)
MERGE: upsert operation
Change Data Feed: row-level change tracking
Liquid Clustering: incremental co-location
Deletion Vectors: fast deletes (no rewrite)
Delta Sharing: cross-org data sharing
Unity Catalog: 3-level governance namespace

### 8. Streaming — Complete Reference
Sources: kafka, rate, socket, file (csv/json/delta)
Sinks:   console, memory, file, kafka, delta,
         foreach, foreachBatch

Output modes:
append:   new rows only (no aggregation)
complete: all rows (with aggregation)
update:   changed rows (with/without agg)

Triggers:
default:           process ASAP
processingTime:    fixed interval
once:              process all then stop
availableNow:      like once but multi-batch
continuous:        experimental, ms latency

Checkpointing: ALWAYS required in production!
Watermark:     .withWatermark("ts", "10 min")
               bounds state + handles late data

### 9. Most Common Cert Trap Questions
Q: What does cache() return?
A: The SAME DataFrame (transformation, lazy!)

Q: difference between count() on DF vs RDD?
A: Same result, but DF uses Catalyst optimization

Q: Can you use append output mode with aggregation?
A: NO — use complete or update!

Q: What happens if you don't set checkpoint?
A: Streaming still works but NO fault tolerance!

Q: Is coalesce(0) valid?
A: NO — minimum 1 partition

Q: What triggers AQE optimization?
A: Shuffle boundaries (exchange operator)

Q: Can you time-travel past VACUUM?
A: NO — vacuumed files are permanently deleted!

Q: Default Delta checkpoint frequency?
A: Every 10 commits to _delta_log/

### 10. Practice Questions — Answer from Memory
1. Write window function for running total
2. Write MERGE statement for upsert
3. Explain predicate pushdown with example
4. When does broadcast join NOT work?
5. What is the difference between
   persist(MEMORY_AND_DISK) and cache()?
6. How do you handle late data in streaming?
7. What triggers a shuffle? List 5 operations.
8. Explain Z-ORDER vs partitioning
9. What does OPTIMIZE do in Delta Lake?
10. How does AQE handle data skew?
