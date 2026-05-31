# Phase 3 LLM Day 3 — Advanced RAG
# Date: May 31, 2026
# Production-grade retrieval + generation!

import numpy as np
from typing import List, Dict, Tuple
import mlflow

print("="*60)
print("Advanced RAG — Production Patterns")
print("="*60)

"""
RAG EVOLUTION — FROM NAIVE TO PRODUCTION

Naive RAG (Phase 1):
  Query → Embed → FAISS search → Top-k → LLM
  Problems:
  → Retrieval quality limited by embedding similarity
  → Fixed chunk size (may split context badly)
  → No reranking (embedding ≠ relevance!)
  → Single query (misses reformulations)
  → No source validation

Advanced RAG addresses ALL of these!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CHUNKING STRATEGIES

Fixed-size (naive):
  text.split(every_N_chars) → equal chunks
  Problem: splits sentences mid-thought!

Recursive character splitting (better):
  Try: paragraph → sentence → word → char
  Split only when needed to fit max_tokens
  Preserves semantic boundaries!

Semantic chunking (best):
  Use embeddings to find natural break points
  Split where cosine similarity drops!
  Most expensive but highest quality.

Sentence window (practical):
  Embed individual sentences
  Retrieve sentence + N neighbors as context
  Best retrieval precision + rich context!

Document hierarchy:
  Summary → Section → Paragraph → Sentence
  Retrieve at multiple levels!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. HYBRID SEARCH

Dense retrieval (embedding):
  + Captures semantic meaning
  + "fast database" finds "quick DB"
  - Misses exact keywords
  - Needs large training data

Sparse retrieval (BM25/TF-IDF):
  + Exact keyword matching
  + Works out of the box
  - Misses semantic similarity
  - No neural understanding

Hybrid = Dense + Sparse (BEST!)
  Reciprocal Rank Fusion (RRF):
  score(d) = Σ 1/(k + rank_dense(d))
           + Σ 1/(k + rank_sparse(d))
  k=60 typical (prevents top ranks dominating)

  Weighted combination:
  score = α * dense_score + (1-α) * sparse_score
  α = 0.5 typical (tune for your data!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. RERANKING (the biggest quality boost!)

Problem: embedding retrieval finds semantically
         similar docs but not most RELEVANT!
"fast" ≈ "quick" ≈ "rapid" ≈ "swift"
→ All retrieved but only one is truly relevant

Reranker: cross-encoder model
  Instead of: encode query, encode doc separately
  Cross-encoder: [CLS] query [SEP] doc → score
  Looks at BOTH together → much better relevance!

Process:
  1. Retrieve top-100 with dense/sparse
     (fast! separate encodings)
  2. Rerank top-100 with cross-encoder
     (slow but worth it for top-10!)
  3. Return top-10 to LLM

Why 2-stage?
  Cross-encoder O(n): 1000 docs = 1000 inferences!
  Dense retrieval O(1): one query embedding!
  2-stage: O(1) + O(100) = practical!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. QUERY TRANSFORMATIONS

HyDE (Hypothetical Document Embeddings):
  1. Ask LLM: "What would a document look like
     that answers this question?"
  2. Embed the HYPOTHETICAL document
  3. Search with that embedding!
  Often retrieves better than raw query!

Multi-query:
  1. Generate 3-5 paraphrases of query
  2. Retrieve for each
  3. Deduplicate + merge results
  Covers different phrasings!

Step-back prompting:
  Abstract query to higher level
  "What is LoRA?" → "What is parameter-efficient
                     fine-tuning?"
  Retrieves more general context first!
"""

# 1. Chunking strategies
class DocumentChunker:
    def __init__(self,
                  chunk_size=512,
                  overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def fixed_size(self,
                    text: str) -> List[str]:
        """Simple fixed-size chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size,
                       len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    def recursive_split(self,
                         text: str) -> List[str]:
        """Split on natural boundaries"""
        separators = ["\n\n", "\n", ". ",
                       " ", ""]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                for part in parts:
                    if len(current) + len(part) \
                            < self.chunk_size:
                        current += part + sep
                    else:
                        if current:
                            chunks.append(
                                current.strip()
                            )
                        current = part + sep
                if current:
                    chunks.append(current.strip())
                if all(len(c) <= self.chunk_size
                        for c in chunks):
                    return [c for c in chunks
                             if c.strip()]
        return [text]

    def sentence_window(self, text: str,
                         window: int = 2
                         ) -> List[Dict]:
        """Each sentence + surrounding context"""
        sentences = text.split('. ')
        results = []
        for i, sent in enumerate(sentences):
            start = max(0, i - window)
            end = min(len(sentences),
                       i + window + 1)
            context = '. '.join(
                sentences[start:end]
            )
            results.append({
                'sentence': sent,
                'context': context,
                'index': i
            })
        return results

# Test chunking
sample_text = """
Apache Spark is a distributed computing engine.
It processes large datasets across clusters.
Delta Lake provides ACID transactions on top.
MLflow tracks machine learning experiments.
LoRA enables efficient LLM fine-tuning.
""" * 5

chunker = DocumentChunker(
    chunk_size=100, overlap=20
)
fixed = chunker.fixed_size(sample_text)
recursive = chunker.recursive_split(sample_text)

print("=== CHUNKING COMPARISON ===")
print(f"Fixed chunks:     {len(fixed)}")
print(f"Recursive chunks: {len(recursive)}")
print(f"\nFirst fixed chunk:\n  {fixed[0][:80]}...")
print(f"\nFirst recursive:\n  {recursive[0][:80]}...")

# 2. BM25 sparse retrieval
class BM25:
    """BM25 sparse retrieval"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.tf = []
        self.df = {}
        self.idf = {}
        self.avgdl = 0

    def fit(self, documents: List[str]):
        self.docs = documents
        N = len(documents)

        # Term frequencies per doc
        for doc in documents:
            words = doc.lower().split()
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
                self.df[w] = \
                    self.df.get(w, 0) + \
                    (1 if w not in tf else 0)
            self.tf.append(tf)

        # Average doc length
        self.avgdl = np.mean([
            len(d.split()) for d in documents
        ])

        # IDF scores
        for word, freq in self.df.items():
            self.idf[word] = np.log(
                (N - freq + 0.5) /
                (freq + 0.5) + 1
            )

    def score(self, query: str,
               doc_idx: int) -> float:
        words = query.lower().split()
        doc = self.docs[doc_idx]
        dl = len(doc.split())
        score = 0
        for word in words:
            if word not in self.tf[doc_idx]:
                continue
            tf = self.tf[doc_idx][word]
            idf = self.idf.get(word, 0)
            # BM25 formula
            score += idf * (
                tf * (self.k1 + 1) /
                (tf + self.k1 * (
                    1 - self.b + self.b *
                    dl / self.avgdl
                ))
            )
        return score

    def retrieve(self, query: str,
                  top_k: int = 5
                  ) -> List[Tuple[int, float]]:
        scores = [
            (i, self.score(query, i))
            for i in range(len(self.docs))
        ]
        return sorted(scores,
                        key=lambda x: x[1],
                        reverse=True)[:top_k]

# Test BM25
docs = [
    "Apache Spark processes large datasets",
    "Delta Lake provides ACID transactions",
    "MLflow tracks experiments and models",
    "LoRA fine-tunes LLMs efficiently",
    "CLIP connects images and text",
    "YOLOv8 detects objects in real-time",
]
bm25 = BM25()
bm25.fit(docs)
results = bm25.retrieve("Spark distributed", 3)

print("\n=== BM25 RETRIEVAL ===")
print("Query: 'Spark distributed'")
for idx, score in results:
    print(f"  [{score:.3f}] {docs[idx]}")

# 3. Reciprocal Rank Fusion
def reciprocal_rank_fusion(
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """Combine dense + sparse rankings"""
    scores = {}

    # Dense contribution
    for rank, (idx, _) in enumerate(
            dense_results):
        scores[idx] = scores.get(idx, 0) + \
            1 / (k + rank + 1)

    # Sparse contribution
    for rank, (idx, _) in enumerate(
            sparse_results):
        scores[idx] = scores.get(idx, 0) + \
            1 / (k + rank + 1)

    # Sort by combined score
    return sorted(scores.items(),
                   key=lambda x: x[1],
                   reverse=True)

# Simulate dense results
dense_results = [(0, 0.95), (2, 0.87),
                  (3, 0.82), (1, 0.78),
                  (4, 0.71)]
sparse_results = bm25.retrieve(
    "Spark distributed", 5
)

hybrid = reciprocal_rank_fusion(
    dense_results, sparse_results
)
print("\n=== HYBRID SEARCH (RRF) ===")
print("Dense + BM25 → RRF fusion:")
for idx, score in hybrid[:3]:
    print(f"  [{score:.4f}] {docs[idx]}")

# 4. Cross-encoder reranker
print("\n=== CROSS-ENCODER RERANKING ===")
print("""
# Production reranking with sentence-transformers

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2'
)

query = "How does LoRA reduce parameters?"
candidates = [
    "LoRA uses low-rank decomposition B×A",
    "Adam optimizer uses moment estimates",
    "LoRA freezes base model weights",
    "QLoRA quantizes to 4-bit NF4",
    "Attention uses Q, K, V projections",
]

# Score all candidates jointly with query!
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# Reranked results
ranked = sorted(
    zip(scores, candidates),
    reverse=True
)
for score, doc in ranked:
    print(f"  [{score:.3f}] {doc}")

# Key: cross-encoder sees BOTH together!
# Much better relevance than bi-encoder!
""")

# 5. Production RAG pipeline
print("\n=== PRODUCTION RAG PIPELINE ===")
print("""
class ProductionRAG:
    def __init__(self):
        # Dense retriever
        self.embedder = SentenceTransformer(
            'BAAI/bge-large-en-v1.5'
        )
        self.vector_db = FAISSIndex()

        # Sparse retriever
        self.bm25 = BM25()

        # Reranker
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )

        # LLM
        self.llm = ChatOpenAI(model="gpt-4")

    def query(self, question: str,
               top_k: int = 5) -> str:
        # Step 1: Multi-query expansion
        queries = self.expand_queries(question)

        # Step 2: Hybrid retrieval
        all_candidates = set()
        for q in queries:
            dense = self.vector_db.search(
                self.embedder.encode(q), 50
            )
            sparse = self.bm25.retrieve(q, 50)
            fused = reciprocal_rank_fusion(
                dense, sparse
            )
            all_candidates.update(
                idx for idx, _ in fused[:20]
            )

        # Step 3: Rerank
        candidates = list(all_candidates)
        pairs = [[question, self.docs[i]]
                  for i in candidates]
        scores = self.reranker.predict(pairs)
        top_idx = np.argsort(scores)[::-1][:top_k]
        context = [self.docs[candidates[i]]
                    for i in top_idx]

        # Step 4: Generate with context
        prompt = self.build_prompt(
            question, context
        )
        return self.llm.predict(prompt)

    def expand_queries(self, q: str
                        ) -> List[str]:
        prompt = f"Generate 3 paraphrases: {q}"
        variants = self.llm.predict(prompt)
        return [q] + variants.split('\\n')[:2]
""")

# 6. RAG evaluation
print("\n=== RAG EVALUATION METRICS ===")
print("""
Retrieval quality:
  Recall@k:    % relevant docs in top-k
  Precision@k: % of top-k that are relevant
  MRR:         mean reciprocal rank of first hit
  NDCG:        normalized discounted cumulative gain

Generation quality:
  Faithfulness: answer grounded in context?
                (no hallucination!)
  Relevance:    answer relevant to question?
  Completeness: all aspects covered?

RAGAS framework (automated evaluation):
  from ragas import evaluate
  from ragas.metrics import (
      faithfulness,
      answer_relevancy,
      context_recall,
      context_precision
  )
  results = evaluate(dataset, metrics=[...])
""")

# 7. Log to MLflow
mlflow.set_experiment("phase3_advanced_rag")
with mlflow.start_run(
        run_name="Advanced_RAG_pipeline"):
    mlflow.log_params({
        "retrieval": "hybrid_BM25_dense",
        "fusion": "RRF_k60",
        "reranker": "cross-encoder",
        "chunking": "recursive_split",
        "chunk_size": 512,
        "chunk_overlap": 50
    })
    mlflow.log_metrics({
        "n_docs": len(docs),
        "n_chunks_fixed": len(fixed),
        "n_chunks_recursive": len(recursive),
    })
    print("\nAdvanced RAG logged to MLflow!")

print("\n" + "="*60)
print("Advanced RAG — MASTERED! 🔍")
print("LLM Week Day 3 COMPLETE!")
print("="*60)
