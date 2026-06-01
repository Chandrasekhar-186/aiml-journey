## LLM Evaluation Summary

Metric selection by task:
Translation:   BLEU (n-gram precision)
Summarization: ROUGE-L (recall)
QA/RAG:        Faithfulness + Relevance
Code:          Pass@k (execution-based!)
General:       LLM-as-judge (most flexible)
Knowledge:     MMLU accuracy

## BLEU Formula
BP × exp(Σ wₙ log pₙ)
BP = brevity penalty (short → penalized)
pₙ = clipped n-gram precision

## ROUGE-N Formula
matched_ngrams / total_ref_ngrams
(recall-oriented — how much ref is covered?)

## Hallucination Types
Factual:    wrong facts (hard to detect)
Faithful:   contradicts source (detect with NLI!)
Intrinsic:  contradicts itself
→ RAG systems must minimize faithful hallucination!

## ReAct Agent Loop
Thought → Action → Observation → repeat
Terminates when: answer found OR max_steps

## MLflow Agent Tracing
@mlflow.trace decorator
mlflow.start_span() context manager
Captures: latency, inputs, outputs per step
View in MLflow UI → Traces tab

## Sliding Window — Max Count Pattern
max_count = max seen in current window
window_size - max_count = replacements needed
If > k: shrink window from left
Key: max_count never decreases (optimization!)
     even when shrinking — still valid!

## Phase 3 Final 3 Days
Day 25: LLM Eval + Agents ✅ today
Day 26: LLM Project
Day 27: Phase 3 complete + stats
Day 28: Phase 4 preview + LeetCode blitz
→ 99 days to application! 🎯
