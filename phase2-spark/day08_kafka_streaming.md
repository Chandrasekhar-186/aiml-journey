# Kafka + Spark Structured Streaming
Date: April 18, 2026

## Kafka Architecture (quick review)
Producer → Topic → Consumer Group
Topic = ordered, immutable log of messages
Partition = unit of parallelism in Kafka
Offset = position of message in partition

## Reading from Kafka in Spark
```python
kafka_df = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers",
            "broker1:9092,broker2:9092")
    .option("subscribe", "ml-predictions")
    .option("startingOffsets", "earliest")
    .load()
)

# Kafka gives raw bytes — deserialize!
parsed = (kafka_df
    .select(
        F.col("key").cast("string"),
        F.col("value").cast("string"),
        F.col("timestamp"),
        F.col("partition"),
        F.col("offset")
    )
    .select(
        F.from_json(
            F.col("value"),
            "model STRING, score DOUBLE,
             ts TIMESTAMP"
        ).alias("data"),
        F.col("timestamp")
    )
    .select("data.*", "timestamp")
)
```

## Writing to Kafka from Spark
```python
(predictions
    .select(
        F.to_json(F.struct("*"))
         .alias("value")
    )
    .writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers",
            "broker1:9092")
    .option("topic", "ml-results")
    .option("checkpointLocation",
            "/tmp/kafka_checkpoint")
    .start()
)
```

## Kafka + Delta Lake Pattern (Lakehouse!)
## Key Options
startingOffsets:  "earliest" or "latest"
maxOffsetsPerTrigger: rate limiting
failOnDataLoss:   false for production
checkpointLocation: ALWAYS set this!

## Checkpoint = Fault Tolerance
Checkpoint stores:
→ Current offsets being processed
→ State for stateful operations
→ Metadata about streaming query

If job crashes → restart from checkpoint
→ Exactly-once processing guaranteed!
