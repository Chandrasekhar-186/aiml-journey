# Day 15 — Attention Mechanism + Transformers
# Date: March 27, 2026
# Foundation of ALL modern LLMs!

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import mlflow

# 1. Self-Attention from scratch!
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq_len, embed = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq, embed)
        K = self.W_k(x)
        V = self.W_v(x)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)
        return self.W_o(output), attention_weights

# 2. Test self-attention
batch_size, seq_len, embed_dim = 2, 10, 64
x = torch.randn(batch_size, seq_len, embed_dim)
attention = SelfAttention(embed_dim, num_heads=8)
output, weights = attention(x)

print(f"Input shape:            {x.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weights shape:{weights.shape}")
print(f"Weights sum to 1:       "
      f"{weights[0,0].sum():.4f}")  # should be ~1.0

# 3. Simple Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(
            embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual connection
        attn_out, _ = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward + residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# 4. Simple text classifier using Transformer
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 num_heads, ff_dim, num_classes,
                 max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(
            max_seq_len, embed_dim)
        self.transformer = TransformerBlock(
            embed_dim, num_heads, ff_dim)
        self.classifier = nn.Linear(
            embed_dim, num_classes)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len)
        # Token + positional embeddings
        x = (self.embedding(x) +
             self.pos_embedding(positions))
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)

# 5. Log architecture to MLflow
mlflow.set_experiment("transformer_architecture")
with mlflow.start_run(run_name="TransformerBlock_v1"):
    mlflow.log_param("embed_dim", 64)
    mlflow.log_param("num_heads", 8)
    mlflow.log_param("ff_dim", 256)
    mlflow.log_param("architecture", "Transformer")

    model = TransformerClassifier(
        vocab_size=10000, embed_dim=64,
        num_heads=8, ff_dim=256, num_classes=2
    )
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    mlflow.log_metric("total_params", total_params)
    print(f"\nTransformer total params: "
          f"{total_params:,}")
    mlflow.pytorch.log_model(model, "transformer")

print("\nKey insight:")
print("Attention allows each token to 'look at'")
print("all other tokens — this is why Transformers")
print("understand context better than RNNs!")
