## ViT Architecture Steps
1. Patch embedding: Conv2d(P, P, stride=P)
   (H×W×C) → N patches, N=(H/P)²
2. Add [CLS] token (prepend to sequence)
3. Add learned position embeddings
4. L × Transformer blocks (pre-norm!)
5. Take [CLS] output → MLP head

## Key Numbers (ViT-B/16)
img_size=224, patch_size=16
N = (224/16)² = 196 patches
embed_dim = 768, depth=12, heads=12
Total: 86M parameters

## ViT Training Requirements
Weight decay: 0.05-0.1 (higher than CNNs!)
LR warmup: first 10% of training
Label smoothing: 0.1
Data augmentation: RandAugment + MixUp + CutMix
Large batch: 4096 typical

## Pre-norm vs Post-norm
Original ViT:   post-norm (less stable)
Modern ViT:     pre-norm (more stable!)
Pre-norm:  x = x + Sublayer(LayerNorm(x))
Post-norm: x = LayerNorm(x + Sublayer(x))

## Attention Map Interpretation
CLS token attention to patches:
→ Shows which image regions model looks at
→ High attention = important for classification
→ Naturally focuses on objects (not background!)
→ Visualize: reshape attn to (H/P, W/P)

## Binary Search on Answer Space
Template:
l, r = min_possible, max_possible
while l < r:
    mid = (l + r) // 2
    if feasible(mid):
        r = mid   # can do better
    else:
        l = mid + 1  # need more

Use when: minimize maximum, find threshold
Examples: Koko, Ship, Split Array Largest Sum

## CV Week Progress
Day 15: Transfer Learning ✅
Day 16: YOLOv8           ✅
Day 17: U-Net            ✅
Day 18: ViT              ✅ today
Day 19: CLIP + multimodal ← tomorrow!
Day 20: CV + Spark
Day 21: CV Project! 🎯
