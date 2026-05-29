# Phase 3 LLM Day 1 — BERT + GPT Deep Dive
# Date: May 29, 2026
# The two paradigms of LLMs!

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModel,
    pipeline
)
import mlflow

print("="*60)
print("BERT + GPT — The Two LLM Paradigms")
print("="*60)

"""
TRANSFORMER VARIANTS — COMPLETE COMPARISON

Original Transformer (Vaswani 2017):
  Encoder + Decoder
  Encoder: bidirectional attention
  Decoder: masked (causal) + cross-attention
  Use: machine translation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BERT (Bidirectional Encoder Representations
      from Transformers, Google 2018):

Architecture: ENCODER ONLY
Attention: BIDIRECTIONAL
  → Each token sees ALL other tokens
  → Left + right context!

Pretraining objectives:
1. Masked Language Model (MLM):
   "The [MASK] sat on the mat"
   → Predict "cat" (15% tokens masked)
   → Forces bidirectional understanding!

2. Next Sentence Prediction (NSP):
   Sentence A + [SEP] + Sentence B
   → Predict: are these consecutive?
   (Later work showed NSP is mostly useless)

Use cases:
✅ Text classification
✅ Named entity recognition
✅ Question answering (extractive)
✅ Sentence similarity
❌ Text generation (not designed for this!)

Variants: RoBERTa, DeBERTa, ALBERT, DistilBERT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPT (Generative Pre-trained Transformer,
     OpenAI 2018):

Architecture: DECODER ONLY
Attention: CAUSAL (left-to-right only)
  → Each token sees only PREVIOUS tokens
  → No peeking at future!

Pretraining objective:
Next Token Prediction (Causal LM):
  "The cat sat on the ___"
  → Predict "mat"
  → Learns from ALL tokens (not just masked!)

Use cases:
✅ Text generation
✅ Few-shot learning (GPT-3+)
✅ Instruction following (ChatGPT)
✅ Code generation (Codex)
❌ Bidirectional tasks (worse than BERT)

Variants: GPT-2, GPT-3, GPT-4, LLaMA,
          Mistral, Falcon, Gemma

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

T5 (Text-to-Text Transfer Transformer,
    Google 2019):

Architecture: ENCODER + DECODER (full!)
Treats EVERYTHING as text→text:
  Classification: "classify: The movie was..."
                  → "positive"
  Translation:    "translate en to fr: Hello"
                  → "Bonjour"
  Summarization:  "summarize: Long article..."
                  → "Short summary"

Pretraining: span corruption (like BERT MLM
             but spans of tokens)

Use cases: any NLP task as text→text
Variants: T5, mT5, FLAN-T5, UL2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAUSAL MASKING — The key difference!

BERT (no mask):
Token 3 can attend to tokens 1,2,3,4,5
→ Full attention matrix

GPT (causal mask):
Token 3 can attend to tokens 1,2,3 ONLY
→ Lower triangular attention matrix

Implementation:
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask==0, -inf)
→ After softmax: future positions = 0!
"""

# 1. Build causal mask
def create_causal_mask(seq_len):
    mask = torch.tril(
        torch.ones(seq_len, seq_len)
    ).bool()
    return mask

mask = create_causal_mask(5)
print("Causal mask (5×5):")
print(mask.int().numpy())
print("Token i can only attend to tokens ≤ i!")

# 2. BERT with HuggingFace
print("\n=== BERT IN ACTION ===")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)
bert = BertModel.from_pretrained(
    'bert-base-uncased'
)
bert.eval()

# Encode text
text = "The Databricks Lakehouse combines "
       "data warehousing and ML"
tokens = tokenizer(
    text,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=128
)

print(f"Text: {text}")
print(f"Tokens: {tokenizer.tokenize(text)}")
print(f"Input IDs shape: "
      f"{tokens['input_ids'].shape}")

with torch.no_grad():
    outputs = bert(**tokens)

# CLS token = sentence embedding!
cls_embedding = outputs.last_hidden_state[:,0,:]
print(f"CLS embedding: {cls_embedding.shape}")
# (1, 768) — 768-dim sentence vector!

# All token embeddings
all_embeddings = outputs.last_hidden_state
print(f"All token embeddings: "
      f"{all_embeddings.shape}")

# 3. Semantic similarity with BERT
print("\n=== BERT SEMANTIC SIMILARITY ===")

def get_embedding(text, tokenizer, model):
    tokens = tokenizer(
        text, return_tensors='pt',
        padding=True, truncation=True,
        max_length=128
    )
    with torch.no_grad():
        out = model(**tokens)
    return out.last_hidden_state[:,0,:]

sentences = [
    "Apache Spark processes big data",
    "Spark is a distributed computing engine",
    "I love eating pizza on weekends",
    "Delta Lake provides ACID transactions",
]

embeddings = torch.cat([
    get_embedding(s, tokenizer, bert)
    for s in sentences
])

# Compute cosine similarity matrix
norm = F.normalize(embeddings, dim=1)
sim_matrix = norm @ norm.T

print("Cosine similarity matrix:")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i < j:
            sim = sim_matrix[i,j].item()
            print(f"  [{sim:.3f}] "
                  f"{s1[:30]} ↔ "
                  f"{s2[:30]}")

# 4. GPT-2 text generation
print("\n=== GPT-2 TEXT GENERATION ===")
gpt_tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2'
)
gpt_model = GPT2LMHeadModel.from_pretrained(
    'gpt2'
)
gpt_tokenizer.pad_token = \
    gpt_tokenizer.eos_token
gpt_model.eval()

prompt = "Apache Spark is a distributed"
inputs = gpt_tokenizer(
    prompt, return_tensors='pt'
)

with torch.no_grad():
    outputs = gpt_model.generate(
        inputs['input_ids'],
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=
            gpt_tokenizer.eos_token_id
    )

generated = gpt_tokenizer.decode(
    outputs[0], skip_special_tokens=True
)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")

# 5. Generation strategies
print("\n=== GENERATION STRATEGIES ===")
print("""
Greedy decoding: always pick highest probability
→ Fast but repetitive

Beam search: keep top-k sequences at each step
→ Better quality, more compute
→ beam_size=4 is common

Top-k sampling: sample from top k tokens
→ More diverse output
→ k=50 typical

Top-p (nucleus) sampling:
→ Sample from smallest set with prob ≥ p
→ p=0.9 typical (cuts off long tail)
→ Most widely used in practice!

Temperature: scales logits before softmax
→ T<1: more confident (less random)
→ T>1: more random (more creative)
→ T=0.7: good balance

Repetition penalty:
→ Penalize recently generated tokens
→ Prevents "I am am am am" loops!
""")

# 6. Special tokens
print("\n=== SPECIAL TOKENS ===")
print("""
BERT special tokens:
[CLS]:  start of sequence (classification!)
[SEP]:  separator between sentences
[MASK]: masked token for MLM pretraining
[PAD]:  padding to equal length
[UNK]:  unknown token

GPT special tokens:
<|endoftext|>: end of document
<|padding|>:   padding

ChatGPT/LLaMA instruction tokens:
<s>:        start of sequence
</s>:       end of sequence
[INST]:     start of instruction
[/INST]:    end of instruction
<<SYS>>:    system prompt
<</SYS>>:   end system prompt

Why special tokens matter:
→ Fine-tuning format MUST match pretraining!
→ Wrong token format → poor performance
→ Check tokenizer docs for each model!
""")

# 7. Log to MLflow
mlflow.set_experiment("phase3_bert_gpt")
with mlflow.start_run(
        run_name="BERT_GPT_comparison"):
    mlflow.log_params({
        "bert_model": "bert-base-uncased",
        "bert_params": "110M",
        "bert_architecture": "encoder_only",
        "gpt_model": "gpt2",
        "gpt_params": "117M",
        "gpt_architecture": "decoder_only",
    })
    # Log similarity scores
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = sim_matrix[i,j].item()
            mlflow.log_metric(
                f"sim_{i}_{j}", sim
            )
    print("\nBERT + GPT logged to MLflow!")

print("\n" + "="*60)
print("BERT + GPT — MASTERED! 🤖")
print("LLM Week Day 1 COMPLETE!")
print("="*60)
