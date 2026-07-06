# STAR Stories — Final Polish
Date: July 5, 2026
(Confirmed behavioral questions from 2026 intel)

## STORY 1: Most Technically Complex System
Question: "Tell me about the most technically
complex system you built. What made it complex
and how did you handle it?"

SITUATION: 
For my 6-month Databricks preparation project,
I designed and built a production-grade real-time
ML model monitoring system that needed to detect
model drift within 60 seconds of it occurring.

TASK:
Build an end-to-end pipeline: ingest streaming
predictions → compute drift metrics →
alert + auto-retrain — all within 1 minute,
at scale, with ACID guarantees.

ACTION:
Three parallel complexity challenges:
1. Streaming: Used Kafka + Spark Structured
   Streaming with exactly-once semantics.
   Chose foreachBatch for MLflow compatibility.

2. Statistical rigor: Implemented KS test +
   PSI from scratch on streaming windows.
   Key insight: watermarks to handle late data
   without unbounded memory growth.

3. Production reliability: Delta Lake for ACID
   writes, checkpoint recovery, Z-ORDER on
   model_id for fast drift queries.

RESULT:
- Drift detection in <1 minute (target met)
- 7 Databricks competencies demonstrated
- Bronze→Silver→Gold lakehouse architecture
- MLflow tracking every drift event
- Zero data loss via exactly-once Kafka

What made it complex: streaming + statistics
+ ACID guarantees + MLflow integration all
simultaneously — each individually manageable,
together requiring careful orchestration.

---

## STORY 2: Technical Disagreement
Question: "Tell me about a time you had a strong
technical disagreement with a colleague."

SITUATION:
During a group project, my teammate wanted to
use a single large monolithic Spark job for our
ML pipeline — reasoning it would be simpler.

TASK:
I believed a modular Delta Lakehouse
(Bronze→Silver→Gold) was architecturally
superior but needed to convince without
damaging collaboration.

ACTION:
Rather than just asserting my preference:
1. I built a quick prototype of BOTH approaches
   in one evening (2 hours each)
2. Measured: query performance, data quality
   detectability, debugging ease, incremental
   update cost
3. Presented data: modular approach was 3x
   faster to debug when Silver layer had issues,
   allowed partial reruns, enabled time travel
4. Acknowledged the valid points: monolith IS
   simpler for small datasets
5. Proposed: use modular for production scale,
   monolith for dev/test environment

RESULT:
- Team adopted modular approach
- When we had a data quality bug, the Silver
  layer isolated it immediately (proved the point)
- Relationship preserved — data beat opinion
- Teammate later said it was the right call

Key learning: build the prototype, show the
data. Technical disagreements resolve fastest
when you remove opinion from the equation.

---

## STORY 3: Refactoring Messy Code
Question: "Have you ever had to refactor a
critical, messy piece of code? What was your
approach and how did you ensure nothing broke?"

SITUATION:
My Phase 1 ML pipeline had grown organically
over 3 weeks into a 600-line notebook with
no tests, hardcoded paths, mixed concerns,
and global state everywhere.

TASK:
Refactor into a production-grade modular system
while keeping all 6 Databricks competency
demonstrations intact and adding CI/CD.

ACTION (systematic 4-step approach):
1. CHARACTERIZE before touching:
   - Wrote integration tests capturing current
     behavior (even the bad parts!)
   - Documented what each section actually did
   - Identified the 3 critical paths

2. EXTRACT without changing behavior:
   - Extracted functions one at a time
   - Ran tests after EACH extraction
   - No behavior changes in this phase

3. RESTRUCTURE:
   - Split into: ingestion.py, features.py,
     training.py, evaluation.py, serving.py
   - Each module testable in isolation
   - Dependency injection (no global state)

4. ADD QUALITY:
   - Unit tests for each module
   - GitHub Actions CI/CD pipeline
   - infer_signature for all MLflow models
   - Type hints + docstrings

RESULT:
- 600-line notebook → 5 focused modules
- Test coverage from 0% → 87%
- CI/CD blocks bad commits automatically
- Next feature addition: 30 min vs 3 hours
- Zero production incidents during refactor

Key insight: the tests written in step 1
are the safety net for all subsequent changes.
Never refactor without a test suite first.

---

## EM ROUND QUESTIONS (from 2026 intel)

Q: "Large-scale data processing challenges?"
A: Real-time ML Monitor — Kafka + Spark
   Streaming, exactly-once semantics,
   watermarks for late data, 7 competencies.
   Key challenge: streaming statistics
   computation (KS test) on rolling windows.

Q: "Pipeline optimization experience?"
A: Phase 2 Spark deep dive — salting for
   data skew, AQE tuning, Photon engine,
   broadcast joins, partition pruning.
   Achieved 11x speedup on benchmark query.

Q: "Team collaboration?"
A: Technical disagreement story above.
   Data > opinion framework works universally.
