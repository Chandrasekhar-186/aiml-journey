## CI/CD for ML Flow
Code push → Unit tests → Lint →
Build → Staging deploy → Integration tests →
Accuracy check → Prod deploy → Notify

## Testing Pyramid for ML
Unit:        test individual functions (fast!)
Integration: test full pipeline small data
Quality:     accuracy >= threshold
Drift:       features within bounds
E2E:         full production simulation

## Databricks Asset Bundles (DABs)
databricks.yml = infrastructure as code
targets: dev / staging / prod
resources: jobs, pipelines, models

Commands:
validate → deploy → run → destroy

## GitHub Actions Key Concepts
on: push/PR trigger
jobs: test → deploy-staging → deploy-prod
needs: job dependency chain
secrets: secure credential storage
if: conditional job execution

## Model Promotion Gates
New accuracy ≥ baseline + min_improvement
New accuracy ≥ absolute minimum
Latency within SLA
No significant drift
All gates pass → auto-promote!

## Bit Manipulation Patterns
n & 1:    extract last bit
n >>= 1:  remove last bit (right shift)
result << 1:  make room for new bit
result |= bit: set bit in result

## Phase 4 Week 1 Progress
Day 1: MLflow serving + A/B    ✅
Day 2: Feature store + monitor ✅
Day 3: Workflows + DLT         ✅
Day 4: CI/CD for ML            ✅ today
Day 5: Week 1 project          ← tomorrow
Day 6: Week 1 review
