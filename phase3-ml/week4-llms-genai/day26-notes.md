## LLM Project Architecture
KB (Delta) → Chunk → TF-IDF + BM25
→ RRF hybrid → LLM → Evaluate
→ MLflow tracking

## Production Extensions
TF-IDF → BGE/CLIP embeddings
Simulated LLM → LoRA fine-tuned model
Local eval → RAGAS framework
Manual deploy → Databricks Model Serving
MLflow run → MLflow tracing (per step)

## Two Pointer Pattern
Move shorter boundary:
→ Moving taller = can only decrease width
  without increasing height (never better!)
→ Moving shorter = might find taller bar
  (could increase area!)
Key insight: maximize min(h[l], h[r]) * width

## Phase 3 Stats
30 days: 7 algorithms + 7 DL + 7 CV + 7 LLM
Projects: 4 complete
Certs:    2 free earned
LeetCode: 163+ total (30+ new in Phase 3)
Streak:   83 days and counting!

## Phase 4 Preview (MLOps focus)
Week 1: MLflow production + model serving
Week 2: Docker + CI/CD + Databricks Workflows
Week 3: Feature stores + A/B testing
Week 4: Monitoring + observability
→ Connects Phase 3 models to production!
