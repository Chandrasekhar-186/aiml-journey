# Behavioral Interview Prep
Date: April 1, 2026
Format: STAR (Situation, Task, Action, Result)

## Story 1 — Problem Solving Under Pressure
Q: "Tell me about a time you solved a 
    difficult technical problem"

S: During my 6-month Databricks prep journey,
   on Day 6, Git push kept failing with
   "rejected — fetch first" error.

T: I needed to fix this without losing
   any of my work or breaking my streak.

A: I diagnosed the conflict (browser uploads
   vs local pushes), ran git pull --rebase,
   established a "local-only" workflow rule,
   and documented it in my notes.

R: Zero Git conflicts since Day 7. Established
   a professional Git workflow used daily.
   Taught me to always pull before pushing.

---

## Story 2 — Learning Something New Quickly
Q: "Tell me about a time you learned a new
    technology rapidly"

S: Started this journey with zero knowledge
   of Apache Spark, MLflow, or LeetCode.

T: Had 6 months to master the entire
   Databricks tech stack for a new grad role.

A: Built a structured daily 3-hour plan.
   Committed to GitHub every day.
   Started with foundations, progressed
   to RAG pipelines and LoRA fine-tuning.

R: By Day 20: 35 LeetCode solved, full RAG
   pipeline built, CNN + YOLOv8 running,
   MLflow tracking every experiment.
   20-day perfect streak — zero missed days.

---

## Story 3 — Ownership & Accountability
Q: "Tell me about a time you took ownership
    of a project end-to-end"

S: Built Wine Quality Classifier as Week 2
   mini-project — self-initiated, no guidance.

T: Wanted to compare RF vs GBM vs Neural Net
   with proper MLflow tracking.

A: Designed experiment from scratch:
   feature engineering, model comparison,
   MLflow logging, result analysis, GitHub push.

R: Clean reproducible experiment — all 3 models
   compared side-by-side in MLflow UI.
   Identified XGBoost as best performer.
   Project now on GitHub as portfolio piece.

---

## Story 4 — Handling Failure
Q: "Tell me about a time you failed
    and what you learned"

S: During Day 11, MLflow multi-model comparison
   was confusing — all runs merged incorrectly.

T: Need to properly isolate each model run
   in separate MLflow experiments.

A: Researched MLflow docs, understood that
   each model needs its OWN start_run() block.
   Rewrote the code with correct isolation.

R: Fixed immediately. Now understand MLflow
   run isolation deeply — can explain it
   in interviews confidently.

---

## Story 5 — Bias for Action
Q: "Tell me about a time you made a decision
    with incomplete information"

S: On Day 12, decided to start Computer Vision
   (CNNs) even though it wasn't in Phase 1 plan.

T: Balance staying on schedule vs pursuing
   a strong interest in CV.

A: Assessed that CV knowledge would strengthen
   Databricks application. Added CNN session
   without dropping any existing tasks.

R: 12 days ahead of CV schedule. YOLOv8
   now part of Phase 5 flagship project.
   Decision proved correct — stronger profile.

---

## 10 Must-Prepare Behavioral Questions
1. Why Databricks specifically?
2. Tell me about yourself (2 min pitch)
3. Why do you want to work on Mosaic AI team?
4. Describe your most impressive project
5. How do you handle ambiguity?
6. Tell me about a time you disagreed with someone
7. How do you prioritize when overwhelmed?
8. What's your biggest technical weakness?
9. Where do you see yourself in 3 years?
10. Do you have questions for us?

## Answer to "Why Databricks?"
"Databricks sits at the intersection of the
two things I'm most passionate about —
large-scale data engineering and AI/ML.
The Lakehouse architecture solves a real
problem I've seen while learning: the gap
between where data lives and where models run.
I've been using MLflow, Spark, and Delta Lake
daily for 20 days — I don't just want to work
at Databricks, I want to contribute to the
platform millions of engineers use every day."
