# Day 16 — RAG Foundations
# Date: March 28, 2026
# Retrieval Augmented Generation — Databricks' key GenAI pattern!

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Simple RAG implementation from scratch!

# 1. Knowledge base (would be Delta Lake in production)
knowledge_base = [
    "Apache Spark is a unified analytics engine for large-scale data processing.",
    "MLflow is an open source platform for managing ML lifecycle including tracking.",
    "Delta Lake provides ACID transactions, scalable metadata handling for data lakes.",
    "Databricks was founded by the creators of Apache Spark at UC Berkeley.",
    "PySpark is the Python API for Apache Spark enabling Python developers to use Spark.",
    "MLflow Model Registry provides a central store to manage full lifecycle of ML models.",
    "Delta Lake supports time travel allowing querying historical versions of data.",
    "Databricks Mosaic AI team works on LLM fine-tuning and model serving infrastructure.",
]

# 2. Create embeddings for knowledge base
print("Building knowledge base embeddings...")
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def embed_text(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(embeddings, p=2, dim=1)

# Embed all documents
kb_embeddings = torch.cat(
    [embed_text(doc) for doc in knowledge_base]
)
print(f"Knowledge base: {len(knowledge_base)} docs")
print(f"Embedding shape: {kb_embeddings.shape}")

# 3. Retrieval function
def retrieve(query, top_k=3):
    query_embedding = embed_text(query)
    # Cosine similarity with all docs
    similarities = torch.cosine_similarity(
        query_embedding,
        kb_embeddings
    )
    # Get top-k most similar
    top_indices = similarities.argsort(
        descending=True
    )[:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "doc": knowledge_base[idx],
            "score": similarities[idx].item()
        })
    return results

# 4. Test RAG retrieval
queries = [
    "How does MLflow help with model management?",
    "What is Delta Lake used for?",
    "Who created Apache Spark?"
]

print("\n" + "="*50)
print("RAG Retrieval Results:")
print("="*50)

for query in queries:
    print(f"\nQuery: '{query}'")
    results = retrieve(query, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r['score']:.4f}")
        print(f"      Doc: {r['doc'][:60]}...")

# 5. Full RAG pipeline (without LLM — just retrieval)
def rag_answer(query):
    # Step 1: Retrieve relevant docs
    docs = retrieve(query, top_k=3)

    # Step 2: Build context (augmentation)
    context = "\n".join(
        [f"- {d['doc']}" for d in docs]
    )

    # Step 3: Build prompt for LLM
    prompt = f"""Answer based on context only.

Context:
{context}

Question: {query}

Answer:"""
    return prompt, docs

print("\n" + "="*50)
print("Full RAG Prompt Example:")
print("="*50)
prompt, docs = rag_answer(
    "What makes Delta Lake special?"
)
print(prompt)
print("\nIn production: send this prompt to")
print("Databricks Foundation Model API!")
