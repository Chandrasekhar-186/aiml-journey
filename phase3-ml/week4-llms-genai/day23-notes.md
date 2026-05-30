## LoRA Core Math
ΔW = B × A
A ∈ ℝ^(r×k): down-projection (random init)
B ∈ ℝ^(d×r): up-projection (zero init!)
Scale: α/r

Forward: h = W₀x + (α/r)·B·A·x
Trainable: r×k + d×r vs d×k full
Saving: ~96% fewer parameters at r=16!

## Why B=0 Initialization?
At start: ΔW = B·A = 0·A = 0
→ Model identical to pretrained at step 0
→ No disruption of pretrained knowledge
→ Gradual adaptation as training proceeds!

## LoRA vs QLoRA Memory
Full FT 7B:    14GB weights + 28GB gradients
LoRA 7B fp16:  14GB (same!) + tiny adapters
QLoRA 7B 4bit: ~5GB + tiny adapters ← magic!

## QLoRA Key Components
4-bit NF4 quantization (normal float 4)
Double quantization (quantize the constants!)
Paged optimizers (CPU offload for Adam states)
Result: 7B model on 1× RTX 3090 (24GB)!

## LoRA Hyperparameters
r=16, alpha=16: good default
target_modules: q_proj + v_proj minimum
Learning rate: 10-100× higher than full FT!
Dropout: 0.05-0.1

## Sliding Window Pattern
Maintain fixed-size window
Add new: window[s[i]] += 1
Remove old: window[s[i-k]] -= 1
Delete if 0 to keep counters clean!
Compare window == target_count → O(1)!

## LLM Week Progress
Day 22: BERT + GPT      ✅
Day 23: LoRA + QLoRA    ✅ today
Day 24: Advanced RAG    ← tomorrow
Day 25: LLM evaluation
Day 26: AI Agents
Day 27: LLM project
Day 28: Phase 3 complete!
