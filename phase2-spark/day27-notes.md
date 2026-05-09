## LRU Cache — Key Insight
OrderedDict: O(1) get/put + maintains order
move_to_end(key): marks as recently used
popitem(last=False): removes least recently used
Perfect combination for O(1) LRU!

## Bidirectional BFS — When to Use
Single BFS: O(b^d) — exponential!
Bidir BFS:  O(b^(d/2)) — square root speedup!
Use when:   known start AND end nodes
            searching for connection path
            word ladder, social network distance

Key: always expand the SMALLER frontier
     → keeps both sides balanced

## Word Ladder — Implicit Graph BFS
Nodes: words
Edges: words differing by 1 character
BFS finds shortest path (minimum changes)
Key: generate all 1-char variants efficiently
     26 × word_length = candidates to check

## Phase 3 Pre-loaded Knowledge Summary
Week 3 (CV) pre-loaded:
→ CNN architecture ✅
→ ResNet + transfer learning ✅
→ YOLOv8 basics ✅
→ OpenCV preprocessing ✅

Week 4 (LLM) pre-loaded:
→ Transformer attention ✅ (built from scratch!)
→ HuggingFace pipelines ✅
→ RAG pipeline ✅ (production-grade!)
→ LoRA concept ✅

Phase 3 = going from 30% → 100% depth!

## CodeSignal Hard Patterns
LRU Cache:     OrderedDict + move_to_end
Word Ladder:   BFS on implicit graph
Median Arrays: Binary search on partition
Serialize Tree: BFS level-order + reconstruct
Bidir BFS:     Two frontiers, expand smaller
