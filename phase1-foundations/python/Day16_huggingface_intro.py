mlenv\Scripts\activate
pip install transformers datasets accelerate sentencepiece

# Day 16 — HuggingFace + Pretrained LLMs
# Date: March 28, 2026
# The library Databricks Mosaic AI uses daily!

import mlflow
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
import torch

print("="*50)
print("HuggingFace — Your AI Superpower!")
print("="*50)

# 1. Pipelines — easiest way to use models
# Sentiment Analysis
print("\n1. Sentiment Analysis Pipeline:")
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
texts = [
    "Databricks is an amazing platform!",
    "This bug is really frustrating.",
    "MLflow makes experiment tracking easy.",
    "I failed the coding interview."
]
results = sentiment(texts)
for text, result in zip(texts, results):
    print(f"  '{text[:40]}...' "
          f"→ {result['label']} "
          f"({result['score']:.4f})")

# 2. Text Generation
print("\n2. Text Generation:")
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_length=50
)
prompt = "Apache Spark is used for"
output = generator(prompt, num_return_sequences=1)
print(f"  Prompt: '{prompt}'")
print(f"  Generated: {output[0]['generated_text']}")

# 3. Zero-shot classification
print("\n3. Zero-shot Classification:")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
sequence = ("The model achieved 95% accuracy "
            "on the test set after fine-tuning.")
candidate_labels = ["machine learning",
                    "sports", "politics",
                    "data engineering"]
result = classifier(sequence, candidate_labels)
print(f"  Text: '{sequence[:50]}...'")
for label, score in zip(result['labels'],
                          result['scores']):
    print(f"  {label:20s}: {score:.4f}")

# 4. Tokenizer deep dive
print("\n4. Tokenizer:")
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
text = "Databricks builds the lakehouse platform"
tokens = tokenizer(text, return_tensors="pt")
print(f"  Input text:  '{text}'")
print(f"  Token IDs:   {tokens['input_ids']}")
print(f"  Tokens:      {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
print(f"  Vocab size:  {tokenizer.vocab_size:,}")

# 5. Embeddings — foundation of RAG!
print("\n5. Text Embeddings (RAG foundation):")
from transformers import AutoModel
import torch.nn.functional as F

embed_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
embed_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_embedding(text):
    inputs = embed_tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(embeddings, p=2, dim=1)

# Compare similarity between sentences
sentences = [
    "Spark processes big data at scale",
    "PySpark handles large datasets efficiently",
    "I love eating pizza on weekends"
]
embeddings = [get_embedding(s) for s in sentences]

print("\n  Cosine similarities:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = torch.cosine_similarity(
            embeddings[i], embeddings[j]
        ).item()
        print(f"  S{i+1} vs S{j+1}: {sim:.4f} — "
              f"{'Similar ✅' if sim > 0.7 else 'Different ❌'}")

# 6. Log to MLflow
mlflow.set_experiment("huggingface_models")
with mlflow.start_run(run_name="HF_embeddings"):
    mlflow.log_param("embed_model",
                     "all-MiniLM-L6-v2")
    mlflow.log_param("embed_dim", 384)
    mlflow.log_metric("num_sentences",
                       len(sentences))
    print("\nHuggingFace experiment logged!")
