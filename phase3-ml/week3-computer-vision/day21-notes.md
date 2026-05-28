## CV Project — 7 Competencies
1. YOLOv8: anchor-free detection
2. Spark: distributed Pandas UDF inference
3. Delta Lake: Bronze→Silver→Gold Lakehouse
4. MLflow: metrics + artifacts + tags
5. Pandas UDF: PyTorch↔Spark bridge
6. Production: partitionBy, cache, idempotent
7. CV metrics: confidence, quality tier

## Key Production Patterns Used
explode_outer(): null-safe array expansion
partitionBy("class_name"): query pruning
cache(): reuse inference results
filter(isNotNull): data quality gate
current_timestamp(): audit trail

## CV Week Summary — 7 Days
Day 1: Transfer learning (EfficientNet)
Day 2: YOLOv8 (backbone+neck+head)
Day 3: U-Net (encoder+decoder+skip)
Day 4: ViT (patches+[CLS]+attention)
Day 5: CLIP (contrastive+zero-shot)
Day 6: CV+Spark (Pandas UDF at scale)
Day 7: CV Project (all 7 combined!)

## Set Matrix Zeroes Pattern
First pass: use matrix edges as flags
Second pass: zero based on flags
Final: handle edge rows/cols
Key: don't corrupt flags while using them!

## Sudoku Box Index
box_idx = (row//3)*3 + col//3
Row 0-2, Col 0-2: box 0
Row 0-2, Col 3-5: box 1
Row 0-2, Col 6-8: box 2
Row 3-5, Col 0-2: box 3
...etc

## Phase 3 Week 3 COMPLETE!
Week 4 starts tomorrow: LLMs + GenAI
Day 22: Transformer deep dive
Day 23: BERT vs GPT vs T5
Day 24: LoRA fine-tuning hands-on!
Day 25: RAG advanced
Day 26: LLM evaluation
Day 27: AI Agents
Day 28: Week 4 project!
