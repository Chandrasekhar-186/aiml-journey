## MLlib Pipeline — Key Components

Transformer: transform(df) → new DataFrame
  → StringIndexer, VectorAssembler, Scaler

Estimator:   fit(df) → Transformer (Model)
  → RandomForestClassifier, LogisticRegression

Pipeline:    chains Estimators + Transformers
  → fit() trains all Estimators sequentially
  → transform() applies all Transformers

PipelineModel: fitted Pipeline
  → can be saved + loaded!
  → production deployment artifact

## CrossValidator vs TrainValidationSplit
CrossValidator:         k-fold CV (more accurate)
TrainValidationSplit:   single train/val split
                        (faster, less data needed)

## TF-IDF for Text
Tokenizer  → split text into words
HashingTF  → convert words to feature vector
             (hashing trick for efficiency!)
IDF        → weight by inverse document frequency
             (common words → lower weight)

## Task Scheduler Greedy Insight
Most frequent task needs most "cooling" slots
Formula: (max_freq - 1) * (n + 1) + max_count
Result:  max(actual_tasks, formula_result)

## Eulerian Path (Itinerary)
Use all edges exactly once
DFS + post-order append → reverse for path
Key: visit lexicographically smallest first
     (use min-heap for adjacency list!)
