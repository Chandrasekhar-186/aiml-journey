# Phase 3 Day 12 — Attention + Transformer
# Date: May 20, 2026
# Foundation of ALL modern AI!

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import mlflow

print("="*60)
print("Attention Mechanism + Transformer")
print("="*60)

"""
ATTENTION MECHANISM — COMPLETE MATH

Problem with RNN/LSTM:
→ Sequential processing (slow!)
→ Fixed-size hidden state = information bottleneck
→ Long-range dependencies still hard
→ Can't parallelize across sequence

Attention solution:
→ Look at ALL positions simultaneously
→ Learn which positions are important
→ No bottleneck — direct connections!
→ Fully parallelizable!

SCALED DOT-PRODUCT ATTENTION:

Attention(Q, K, V) = softmax(QKᵀ/√d_k) V

Where:
  Q = Queries  (what am I looking for?)
  K = Keys     (what do I have to offer?)
  V = Values   (what information to retrieve?)
  d_k = key dimension (scaling factor)

Step by step:
1. QKᵀ: compute similarity between every
        query-key pair → attention scores
2. /√d_k: scale to prevent softmax saturation
           (dot products grow large for big d_k!)
3. softmax: normalize to probabilities [0,1]
4. * V: weighted sum of values

WHY √d_k scaling?
If d_k is large: dot products have large variance
→ Softmax pushes toward 0/1 (near one-hot)
→ Gradients vanish!
→ Divide by √d_k: variance stays ~1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MULTI-HEAD ATTENTION:

Instead of one attention, use H heads!
Each head learns different relationships:
  Head 1: subject-verb agreement
  Head 2: pronoun reference
  Head 3: syntactic structure
  ...

MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O
head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)

Benefits:
→ Each head attends to different positions
→ Different subspaces of information
→ Richer representations!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRANSFORMER ARCHITECTURE:

Encoder layer (BERT-style):
  MultiHeadAttention → Add&Norm →
  FeedForward → Add&Norm

Decoder layer (GPT-style):
  Masked MultiHeadAttention → Add&Norm →
  Cross-Attention → Add&Norm →
  FeedForward → Add&Norm

POSITIONAL ENCODING:
Attention has no notion of order!
→ Add positional signal to embeddings

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Why sinusoidal?
→ Unique encoding for each position
→ Relative positions can be computed
→ Generalizes to unseen sequence lengths
→ Learned PE also works (used in BERT)

ADD & NORM (residual + layer norm):
x = LayerNorm(x + Sublayer(x))
→ Residual: like ResNet, prevents grad vanishing
→ LayerNorm: normalize per sample (not batch!)
"""

# 1. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        # Q: (B, H, T, d_k)
        # K: (B, H, T, d_k)
        # V: (B, H, T, d_v)

        # Step 1: Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / self.scale

        # Step 2: Apply mask (for decoder!)
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0, float('-inf')
            )

        # Step 3: Softmax → attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Step 4: Weighted sum of values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(
            self.d_k
        )

    def split_heads(self, x, B):
        """(B, T, d_model) → (B, H, T, d_k)"""
        x = x.view(B, -1,
                     self.num_heads,
                     self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)

        # Project + split into heads
        Q = self.split_heads(self.W_Q(Q), B)
        K = self.split_heads(self.W_K(K), B)
        V = self.split_heads(self.W_V(V), B)

        # Attention per head
        x, attn = self.attention(Q, K, V, mask)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1, self.d_model)

        # Final projection
        return self.W_O(x), attn

# 3. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000,
                  dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len
        ).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # even dims
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # odd dims

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 4. Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff,
                  dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU not ReLU!
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 5. Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads,
                  d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, num_heads
        )
        self.ff = FeedForward(d_model, d_ff,
                               dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + Add & Norm
        attn_out, _ = self.attention(
            x, x, x, mask  # Q=K=V=x for self-attn
        )
        x = self.norm1(
            x + self.dropout(attn_out)
        )

        # Feed-forward + Add & Norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# 6. Complete Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model,
                  num_heads, d_ff,
                  num_layers, num_classes,
                  max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )
        self.pos_encoding = PositionalEncoding(
            d_model, max_len, dropout
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads,
                d_ff, dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # CLS token classification head
        self.classifier = nn.Linear(
            d_model, num_classes
        )

    def forward(self, x, mask=None):
        # Embed + positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Use [CLS] token (first position)
        cls_output = x[:, 0, :]
        return self.classifier(cls_output)

# 7. Test the full transformer
print("\n=== TESTING TRANSFORMER ===")
vocab_size = 1000
d_model = 128
num_heads = 8
d_ff = 512
num_layers = 4
num_classes = 5
seq_len = 50
batch_size = 16

model = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    num_classes=num_classes
)

x = torch.randint(0, vocab_size,
                   (batch_size, seq_len))
output = model(x)

total_params = sum(
    p.numel() for p in model.parameters()
)
print(f"Input shape:    {x.shape}")
print(f"Output shape:   {output.shape}")
print(f"Total params:   {total_params:,}")
print(f"d_model:        {d_model}")
print(f"num_heads:      {num_heads}")
print(f"d_k per head:   {d_model//num_heads}")

# 8. Attention visualization
print("\n=== ATTENTION VISUALIZATION ===")
model.eval()
with torch.no_grad():
    embeddings = model.embedding(x)
    embeddings = model.pos_encoding(embeddings)
    layer0 = model.layers[0]
    _, attn_weights = layer0.attention(
        embeddings, embeddings, embeddings
    )
    print(f"Attention weights shape: "
          f"{attn_weights.shape}")
    print(f"(batch={batch_size}, "
          f"heads={num_heads}, "
          f"seq={seq_len}, seq={seq_len})")
    print(f"Sum across seq (should be 1.0): "
          f"{attn_weights[0,0,0].sum():.4f}")

# 9. Connection to CV week!
print("\n=== ATTENTION IN COMPUTER VISION ===")
print("""
Vision Transformer (ViT):
→ Split image into 16×16 patches
→ Flatten each patch → token embedding
→ Add [CLS] token + positional encoding
→ Run through Transformer encoder
→ Classify from [CLS] token

"An image is worth 16×16 words" (2020 paper)

CLIP (Contrastive Language-Image Pre-training):
→ Text encoder: Transformer
→ Image encoder: ViT or ResNet
→ Contrastive loss: match image ↔ text
→ Zero-shot: encode text query, find images!

Cross-attention (Stable Diffusion, DETR):
→ Q from image features
→ K, V from text/query embeddings
→ Conditions image generation on text!

YOLOv8 neck uses attention too:
→ C2f module = CSP + attention
→ Helps detect multi-scale objects

THIS is why attention matters for CV!
Everything in week 3 builds on today.
""")

# Log to MLflow
mlflow.set_experiment("phase3_attention")
with mlflow.start_run(
        run_name="transformer_encoder"):
    mlflow.log_param("d_model", d_model)
    mlflow.log_param("num_heads", num_heads)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("d_ff", d_ff)
    mlflow.log_metric("total_params",
                       total_params)
    print("\nTransformer logged to MLflow!")

print("\n" + "="*60)
print("Attention + Transformer — MASTERED! 🤖")
print("2 DAYS TO COMPUTER VISION WEEK! 🎨")
print("="*60)
