## CLIP Core Math
Image encoder → d-dim embedding (normalized)
Text encoder  → d-dim embedding (normalized)
Similarity = cos_sim(img, txt)
Loss = InfoNCE (symmetric cross-entropy)
       maximize diagonal of N×N sim matrix

## Temperature (τ) in CLIP
Low τ:  sharp distribution (confident)
High τ: soft distribution (uncertain)
CLIP learns τ during training!
Initial τ = 1/0.07 ≈ 14.3

## Zero-Shot Classification
1. Encode text prompts: "a photo of a {class}"
2. Encode query image
3. Cosine similarity → argmax = prediction
No fine-tuning needed! Any classes work!

## Prompt Engineering for CLIP
"a photo of a {class}" → general objects
"a satellite photo of {class}" → aerial
Context matters! Experiment with MLflow!

## Databricks Intel — 5 Key Problems
TicTacToe:    row/col/diag sum tracking O(1)
TimeMap:      binary search on timestamps
SnapshotArray: binary search on snap_ids
IncrEncode:   traverse BACKWARD (avoid overwrite!)
LazyArray:    accumulate ops list, execute at indexOf

## Common Pattern: Binary Search on State
All three data structure problems use:
→ Store (timestamp/snap_id, value) pairs
→ Binary search to find latest ≤ query
Template: find rightmost value ≤ target
