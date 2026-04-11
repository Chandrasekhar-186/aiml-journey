## Monotonic Stack Pattern
Stack maintains increasing OR decreasing order
When violated → pop and process

Next Greater Element pattern:
for i, val in enumerate(arr):
    while stack and val > arr[stack[-1]]:
        idx = stack.pop()
        result[idx] = i - idx  # or val - arr[idx]
    stack.append(i)

Applications:
→ Daily temperatures (next warmer day)
→ Largest rectangle in histogram
→ Next greater element
→ Stock span problem

## Two Heap Pattern (Median Stream)
lower = max-heap (store negated in Python)
upper = min-heap
Invariant: len(lower) == len(upper) OR
           len(lower) == len(upper) + 1
Median: lower[0] if odd, avg of both if even

## Phase 1 DSA Pattern Summary
Array/String: HashMap, Two Pointer, Sliding Win
Tree:         DFS recursive, BFS queue
Graph:        DFS (visited set), BFS (queue),
              Union-Find, Topological Sort
DP:           1D dp[], 2D dp[][], Set-based
Heap:         Two heaps, Top-K, K-th element
Stack:        Monotonic, matching brackets
Trie:         Prefix tree, word problems
Greedy:       Sort first, local optimal

## Superday Prep Checklist
✅ Coding: 49 problems, patterns memorized
✅ ML Depth: algorithms, evaluation, GenAI
✅ System Design: Lakehouse framework
✅ Behavioral: 8 STAR stories ready
✅ Questions for interviewer: 3 prepared
