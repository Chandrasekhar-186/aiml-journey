## RAG Evolution
Naive: embed → search → generate
Advanced: chunk → hybrid search →
          rerank → multi-query → generate

## Chunking Strategy Selection
Fixed size:      fast, poor quality
Recursive:       balanced ← default choice
Sentence window: best precision
Semantic:        best quality, slow

## Hybrid Search
Dense (embedding): semantic similarity
Sparse (BM25):     exact keyword match
RRF fusion:        best of both worlds!

RRF score = Σ 1/(k + rank_i)
k=60 prevents top ranks dominating

## Cross-Encoder vs Bi-Encoder
Bi-encoder:    encode query + doc separately
               O(1) per query at search time
               Good for retrieval (fast!)

Cross-encoder: encode query + doc TOGETHER
               O(n) per query × n candidates
               Much better relevance scoring
               Use for reranking top-100→10!

## 2-Stage Retrieval Pattern
Stage 1: Bi-encoder → top-100 fast
Stage 2: Cross-encoder → top-10 accurate
Practical: O(1) + O(100) = feasible!

## RAG Evaluation (RAGAS)
Faithfulness:   answer ⊆ context (no hallucination)
Answer Relevancy: answer relevant to question
Context Recall: retrieved relevant docs?
Context Precision: retrieved only relevant?

## Minimum Window Pattern
need = Counter(t), missing = len(t)
Right expands: if need[char]>0: missing-=1
               need[char] -= 1
When missing==0: shrink left
                 track best window
                 advance left by 1
Key: missing tracks WHICH chars still needed!
