## Attention Mechanism — Plain English

Each token computes:
Q (Query)  = "what am I looking for?"
K (Key)    = "what do I contain?"
V (Value)  = "what do I give if selected?"

Attention(Q,K,V) = softmax(QK^T / √d_k) * V

Why divide by √d_k?
→ Prevents dot products from getting too large
→ Keeps softmax in stable gradient range

## Transformer vs RNN
RNN:         processes tokens sequentially (slow)
Transformer: processes ALL tokens in parallel (fast)
→ This is why GPT-4 can be trained on clusters!

## Delta Lake — 4 Key Operations
INSERT → append new rows
UPDATE → modify existing rows
DELETE → remove rows
MERGE  → upsert (insert + update in one!)

## Delta Lake Superpowers
ACID transactions  → no corrupt data ever
Time travel       → query any past version
Schema evolution  → add columns safely
Z-Order indexing  → faster queries on columns
OPTIMIZE          → compact small files

## Binary Search on Rotated Array
Key insight: one half is ALWAYS sorted
If nums[mid] > nums[right]:
    min is in RIGHT half → left = mid + 1
Else:
    min is in LEFT half → right = mid
