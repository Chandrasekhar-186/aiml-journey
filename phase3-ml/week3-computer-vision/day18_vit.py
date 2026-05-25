# Phase 3 CV Day 4 — Vision Transformer (ViT)
# Date: May 25, 2026
# "An Image is Worth 16x16 Words" (2021)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mlflow

print("="*60)
print("Vision Transformer (ViT) from Scratch")
print("="*60)

"""
VIT — COMPLETE MATH

Key paper: "An Image is Worth 16x16 Words"
           (Dosovitskiy et al., 2021, Google)

Core insight: treat image patches as tokens!
→ Split 224×224 image into 14×14 = 196 patches
→ Each patch (16×16×3) = 768-dim vector
→ Process with standard Transformer!

ARCHITECTURE:

Step 1: Patch Embedding
  Image (H×W×C) → N patches of (P×P×C)
  N = (H×W) / (P×P) = (224×224)/(16×16) = 196
  Each patch → linear projection → D dimensions

Step 2: Add [CLS] token
  Prepend learnable classification token
  → Final sequence: N+1 tokens
  → [CLS] aggregates global image info

Step 3: Add Position Embeddings
  Learned 1D positional embeddings
  → Patch positions encoded
  → Without: model can't distinguish patch order!

Step 4: Transformer Encoder (L layers)
  Standard Multi-Head Self-Attention + FFN
  Each patch attends to ALL other patches!
  → Global receptive field from layer 1!

Step 5: Classification Head
  Take [CLS] token output (position 0)
  → MLP → num_classes

COMPARISON WITH CNN:
CNN:  local receptive field → grows slowly
      excellent inductive bias (locality, equivariance)
      good with small data
      translation equivariant by design

ViT:  global receptive field from layer 1
      no inductive bias (learns it!)
      needs large data (JFT-300M to shine)
      or strong augmentation (DeiT!)
      can capture long-range dependencies

Winner: it depends!
  Small dataset (<10K): ResNet/EfficientNet wins
  Large dataset (>100K): ViT wins or ties
  Very large (>1M): ViT dominates

VARIANTS:
ViT-Ti (Tiny):    5M params
ViT-S (Small):    22M params
ViT-B (Base):     86M params  ← standard
ViT-L (Large):    307M params
ViT-H (Huge):     632M params
ViT-G/14 (Giant): 1.8B params

DeiT: Data-efficient ViT
→ Trains on ImageNet only (no JFT!)
→ Distillation token + knowledge distillation
→ Makes ViT practical without giant datasets!
"""

# 1. Patch Embedding
class PatchEmbedding(nn.Module):
    """Split image into patches + linear embed"""
    def __init__(self, img_size=224,
                  patch_size=16, in_channels=3,
                  embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size)**2

        # Single conv = patch extraction + embedding!
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        # → (B, embed_dim, H/P, W/P)
        x = x.flatten(2)
        # → (B, embed_dim, N_patches)
        x = x.transpose(1, 2)
        # → (B, N_patches, embed_dim)
        return x

# 2. Multi-Head Self-Attention (vision)
class ViTAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,
                  dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            embed_dim, embed_dim * 3,
            bias=False
        )
        self.proj = nn.Linear(embed_dim,
                               embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        # Compute Q, K, V in one step
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads,
            self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Aggregate values
        x = (attn @ v).transpose(1,2).reshape(
            B, N, C
        )
        return self.proj(x), attn

# 3. ViT Encoder Block
class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,
                  mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViTAttention(
            embed_dim, num_heads, dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm (modern ViT style)
        # Note: original paper used post-norm
        attn_out, attn_weights = self.attn(
            self.norm1(x)
        )
        x = x + attn_out  # residual!
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

# 4. Complete Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224,
                  patch_size=16, in_channels=3,
                  num_classes=1000, embed_dim=768,
                  depth=12, num_heads=12,
                  mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size,
            in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # [CLS] token (learnable!)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Position embeddings (learnable!)
        # +1 for [CLS] token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches+1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads,
                      mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim,
                               num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(
            self.pos_embed, std=0.02
        )
        nn.init.trunc_normal_(
            self.cls_token, std=0.02
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(
                    m.weight, std=0.02
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        B = x.shape[0]

        # Step 1: Patch embedding
        x = self.patch_embed(x)
        # x: (B, N, embed_dim)

        # Step 2: Prepend [CLS] token
        cls_tokens = repeat(
            self.cls_token, '1 1 d -> b 1 d', b=B
        )
        x = torch.cat([cls_tokens, x], dim=1)
        # x: (B, N+1, embed_dim)

        # Step 3: Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Step 4: Transformer blocks
        attn_weights_all = []
        for block in self.blocks:
            x, attn_w = block(x)
            attn_weights_all.append(attn_w)

        x = self.norm(x)

        # Step 5: [CLS] token → classify
        cls_output = x[:, 0]  # first token!
        logits = self.head(cls_output)

        if return_attn:
            return logits, attn_weights_all
        return logits

# 5. Test ViT
print("\n=== VIT-TINY FOR CIFAR-10 ===")
# Small ViT for CIFAR-10 (32×32 images)
vit = VisionTransformer(
    img_size=32,
    patch_size=4,    # 8×8 = 64 patches
    in_channels=3,
    num_classes=10,
    embed_dim=192,   # Tiny
    depth=6,
    num_heads=3,
    mlp_ratio=4.0,
    dropout=0.1
)

x_test = torch.randn(4, 3, 32, 32)
logits = vit(x_test)
print(f"Input:     {x_test.shape}")
print(f"N patches: {vit.patch_embed.n_patches}")
print(f"Sequence:  {vit.patch_embed.n_patches+1}"
      f" (patches + [CLS])")
print(f"Output:    {logits.shape}")

total = sum(
    p.numel() for p in vit.parameters()
)
print(f"Parameters: {total:,}")

# 6. Attention visualization
print("\n=== ATTENTION VISUALIZATION ===")
logits, attn_list = vit(x_test,
                          return_attn=True)
last_attn = attn_list[-1]
print(f"Attention shape: {last_attn.shape}")
print(f"(batch, heads, seq, seq)")

# [CLS] attention to patches
cls_attn = last_attn[0, :, 0, 1:]
# → (heads, n_patches)
print(f"CLS→patches attention: {cls_attn.shape}")
print("High values = important patches!")
top_patches = cls_attn.mean(0).topk(5)
print(f"Top 5 attended patches: "
      f"{top_patches.indices.tolist()}")

# 7. CIFAR-10 training
print("\n=== TRAINING VITI ON CIFAR-10 ===")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )
])
train_set = torchvision.datasets.CIFAR10(
    './data', train=True, download=True,
    transform=transform
)
subset = torch.utils.data.Subset(
    train_set, range(1000)
)
loader = DataLoader(subset, batch_size=32,
                     shuffle=True)

optimizer = torch.optim.AdamW(
    vit.parameters(), lr=3e-4,
    weight_decay=0.05  # ViT needs strong WD!
)
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

mlflow.set_experiment("phase3_vit")
with mlflow.start_run(run_name="ViT_CIFAR10"):
    mlflow.log_params({
        "model": "ViT-Tiny",
        "patch_size": 4,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 3,
        "img_size": 32,
        "weight_decay": 0.05,
        "label_smoothing": 0.1
    })

    for epoch in range(5):
        vit.train()
        total_loss = correct = total = 0
        for X, y in loader:
            optimizer.zero_grad()
            out = vit(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(
                vit.parameters(), 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        acc = correct/total
        avg_loss = total_loss/len(loader)
        mlflow.log_metrics({
            "loss": avg_loss,
            "accuracy": acc
        }, step=epoch)
        print(f"Epoch {epoch+1}: "
              f"loss={avg_loss:.4f} "
              f"acc={acc:.4f}")

    print("\nViT logged to MLflow!")

# 8. ViT vs CNN comparison
print("\n=== VIT vs CNN ===")
print("""
                    CNN         ViT
Inductive bias:     Strong      None (learns it!)
Small data:         Excellent   Poor (needs pretraining)
Large data:         Good        Excellent
Global attention:   Layer 10+   Layer 1!
Positional info:    Built-in    Must be learned
Computation:        O(H*W)      O((H*W)²/P²)
Interpretability:   Moderate    High (attention maps!)
Transfer learning:  Excellent   Excellent

Best practice 2024:
→ Small dataset: EfficientNet or ConvNeXt
→ Large dataset: ViT-B/16 pretrained
→ Production: depends on latency requirements
→ Research: ViT variants dominate papers
""")

print("\n" + "="*60)
print("Vision Transformer — MASTERED! 🤖")
print("CV Week Day 4 — COMPLETE!")
print("="*60)
