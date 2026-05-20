## Attention Core Formula
Attention(Q,K,V) = softmax(QKᵀ/√d_k) V

Q: what I'm looking for
K: what each position offers
V: information to retrieve
√d_k: prevents softmax saturation

## Why Scale by √d_k?
Large d_k → large dot products
→ softmax saturates (near one-hot)
→ gradients vanish
/√d_k keeps variance ≈ 1 → stable gradients

## Multi-Head Attention
Each head: d_model/H dimensional subspace
Different heads → different relationships
Concat + project → rich representations

## Transformer Block
Self-attention → Add&Norm → FFN → Add&Norm
LayerNorm(x + Sublayer(x))
Residual: gradient highway (like ResNet!)
LayerNorm: normalize per-sample (not batch)

## Positional Encoding
PE(pos,2i) = sin(pos/10000^(2i/d))
PE(pos,2i+1) = cos(pos/10000^(2i/d))
Unique per position, relative distances preserved

## GELU vs ReLU (Transformers use GELU!)
ReLU: hard 0 for z<0
GELU: smooth, probabilistic zeroing
      x * Φ(x) where Φ = normal CDF
Better for NLP/Transformer tasks!

## ViT Connection (CV Week!)
Image → 16×16 patches → flatten → embed
Add [CLS] token + positional encoding
Transformer encoder → classify from [CLS]
Same code as today! Just different input!

## Two Pointer Gap
fast n+1 steps ahead of slow
Both move together → when fast=None
slow is just before Nth from end
Works for: remove Nth, find middle, cycle
