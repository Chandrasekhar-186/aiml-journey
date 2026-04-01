## RAG Production Stack
Text → Chunker → Embeddings → FAISS index
Query → Embed → Similarity search → Top-k docs
Top-k docs + Query → LLM prompt → Answer

## LangChain Key Components
TextSplitter  → chunk documents
Embeddings    → convert text to vectors
VectorStore   → store + search vectors (FAISS)
Retriever     → interface to vector store
Chain         → connect retriever + LLM

## LoRA — Key Numbers to Remember
Full fine-tune: update ALL weights (billions!)
LoRA rank=8:   update ~0.06% of weights
Memory saving: 3-8× less GPU RAM
Quality:       95-99% of full fine-tune quality

LoRA formula: W_new = W_frozen + B @ A
A: (rank × input_dim)  — small!
B: (output_dim × rank) — small!

## Backtracking Template
def backtrack(state):
    if complete: results.append(state); return
    for choice in choices:
        make(choice)      # add to state
        backtrack(state)  # recurse
        undo(choice)      # remove from state

Use for: permutations, combinations,
         subsets, sudoku, N-Queens

## Hard Sliding Window (Min Window)
Two frequency maps: need{} + have{}
formed counter tracks satisfied chars
Expand right → try shrink left
Only shrink when all chars satisfied
