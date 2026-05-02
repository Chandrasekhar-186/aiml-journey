## Cert Simulation 3 — Key Insights

Scenario questions test DEPTH:
→ Don't just state the answer
→ Explain WHY (mechanism matters!)
→ Mention config names exactly
→ Connect to Databricks tools

## Partition Size Formula
Target partition size: ~128MB
Required partitions = dataset_size / 128MB
Example: 100GB / 128MB = ~800 partitions
Default 200 WAY too few for large datasets!
AQE fixes this automatically in Spark 3.2+

## Exactly-Once Streaming
Kafka alone: at-least-once
Spark checkpoint: at-least-once offset tracking
Delta Lake sink: exactly-once (ACID!)
All three together: end-to-end exactly-once

## Greedy Range Tracking Pattern
Instead of exact value: track (min, max) range
Expand range for flexible choices ('*' → ±1)
Contract range for forced choices ('(' or ')')
Valid if max_range >= 0 AND min_range == 0 at end

## Project Final Stats
Lines of code:     ~500
Files:             8 Python + 3 Markdown
Competencies:      7 (Spark, Delta, MLflow,
                   GraphFrames, Streaming,
                   SQL, Production patterns)
Architecture:      Bronze→Silver→Gold→MLflow
Daily GitHub:      50 consecutive commits 🟩

## 11 Days to Exam — Final Checklist
□ Quick-fire 20 traps: answer in <20 seconds each
□ Scenario questions: explain mechanism not just answer
□ Config names: memorize exact parameter names
□ Delta behaviors: exact, not approximate
□ Streaming modes: know which combos INVALID
