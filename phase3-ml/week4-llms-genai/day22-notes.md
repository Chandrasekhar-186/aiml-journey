## BERT vs GPT vs T5 — One-Line Summary
BERT:  encoder-only, bidirectional → understand
GPT:   decoder-only, causal → generate
T5:    encoder+decoder → text-to-text anything

## Pretraining Objectives
BERT: MLM (mask 15%, predict masked tokens)
GPT:  CLM (predict next token always)
T5:   Span corruption (mask spans)

## Attention Mask Types
Full (BERT): all tokens attend to all
Causal (GPT): token i only sees tokens ≤ i
              implemented as lower-triangular mask
Prefix (T5 decoder): encoder attends full,
                     decoder attends causal

## Key Special Tokens
BERT: [CLS] [SEP] [MASK] [PAD] [UNK]
GPT:  <|endoftext|>
LLaMA: <s> </s> [INST] [/INST]

## CLS Token = Sentence Embedding
BERT's [CLS] token at position 0
After all 12 transformer layers:
→ Aggregates info from ALL tokens
→768-dim vector = sentence representation
→ Fine-tune linear head on top for tasks!

## Generation Strategies
Greedy: argmax (fast, repetitive)
Beam:   top-k sequences (good quality)
Top-k:  sample from k most likely
Top-p:  sample from prob mass ≥ p (best!)
Temp:   scale confidence (T<1=focused)

## Bucket Sort Pattern (O(n))
freq[count[num]].append(num)
Iterate freq from high to low
Return first k elements found
