## Phase 2 — Final Mental Models

1. Every Spark job:
   Lazy plan → Action → DAG → Stages → Tasks

2. Every shuffle:
   Map-side partial → Network → Reduce-side final

3. Every Delta write:
   Parquet files + _delta_log/ JSON commit

4. Every streaming query:
   Source → Transform → Watermark → Sink
   + Checkpoint for fault tolerance

5. Every optimization:
   Less data + less shuffles + less scans
   = faster job

## Unbounded DP Pattern
dp[0] = 1 (base: empty)
for i in range(1, target+1):
    for option in options:
        if i >= option:
            dp[i] += dp[i-option]
return sum(dp[low:high+1])

Works for: coin change ways,
           good strings, climbing stairs
Key: at each position, try all options!

## Phase 3 Pre-loaded Knowledge
PyTorch: tensors + autograd + training loop ✅
CNN: conv+pool+flatten+classify ✅
Transfer Learning: freeze+retrain head ✅
Attention: Q/K/V + scaled dot product ✅
RAG: embed+store+retrieve+augment ✅
LoRA: low-rank adaptation concept ✅
HuggingFace: pipelines + tokenizer ✅
YOLOv8: backbone+neck+head+NMS ✅

Phase 3 will go MUCH DEEPER on all of these!
CV week will be particularly exciting 🎯
