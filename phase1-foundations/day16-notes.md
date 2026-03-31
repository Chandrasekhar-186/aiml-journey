## HuggingFace — Key Classes

Pipeline      → easiest API, one line inference
AutoTokenizer → converts text to token IDs
AutoModel     → loads pretrained model weights
Trainer       → fine-tuning helper class
Dataset       → efficient data loading

## RAG — 3 Steps
1. RETRIEVE: embed query → find similar docs
2. AUGMENT:  add docs to prompt as context
3. GENERATE: LLM answers using context only

## Why RAG beats fine-tuning for knowledge:
Fine-tuning: expensive, knowledge becomes stale
RAG:         cheap, knowledge always up-to-date
             just update the vector DB!

## Embeddings — intuition
Similar meaning → similar vector direction
→ cosine similarity close to 1.0
Different meaning → different vector direction
→ cosine similarity close to 0.0

Used in: RAG, semantic search, recommendations

## Sliding Window with Frequency
Pattern: maintain max_count of dominant char
Key insight: window is valid when
(window_size - max_count) <= k
Only need to track MAX count — not update on shrink!
