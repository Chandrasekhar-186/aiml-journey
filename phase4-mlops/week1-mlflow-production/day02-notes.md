## Feature Store Benefits
Training-serving consistency: SAME features!
Point-in-time lookup: no data leakage
Lineage tracking: know feature origins
Reuse: one computation, many models

## Databricks Feature Store
Delta Lake backed (ACID + time travel!)
fs = FeatureStoreClient()
fs.create_table(name, df, primary_keys)
fs.create_training_set(observations,
                        feature_lookups,
                        label)
Model logged WITH feature metadata!

## Drift Detection Tests
KS test:    continuous, p<0.05 = drift
PSI:        < 0.1 = OK
            0.1-0.2 = slight
            > 0.2 = significant drift!
Chi-square: categorical features
Jensen-Shannon: both types, [0,1]

## Model Monitoring Signals
Performance: accuracy drops below threshold
Trend: consistent week-over-week decline
Prediction: distribution shift in outputs
Input: feature drift (use KS + PSI)
Label: class balance change

## Alerting Strategy
P0 (immediate): accuracy < 0.7
P1 (24 hours):  accuracy < threshold
P2 (weekly):    drift in top features
Action: retrain → validate → promote

## Floyd's as Problem Locator
Phase 1: fast+slow meet somewhere in cycle
Phase 2: reset slow to start
         both advance at speed 1
         meet at CYCLE START
Same logic: drift "started" at specific point!
