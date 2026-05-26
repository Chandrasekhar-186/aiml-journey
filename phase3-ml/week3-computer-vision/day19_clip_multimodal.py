# Phase 3 CV Day 5 — CLIP + Multimodal AI
# Date: May 26, 2026
# Zero-shot vision via language supervision!

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mlflow

print("="*60)
print("CLIP — Contrastive Language-Image Pre-training")
print("="*60)

"""
CLIP — COMPLETE UNDERSTANDING

Paper: "Learning Transferable Visual Models
        From Natural Language Supervision"
        (Radford et al., OpenAI, 2021)

Core idea: learn image + text embeddings
           in the SAME space!
           → Similar image+text → close vectors
           → Dissimilar → far apart

TRAINING:
Given N (image, text) pairs:

1. Image encoder: image → image_embedding
   (ResNet or ViT backbone)

2. Text encoder: text → text_embedding
   (Transformer)

3. Both projected to shared d-dim space

4. Contrastive loss (InfoNCE):
   For N pairs, creates N×N similarity matrix
   Diagonal = matching pairs (positive!)
   Off-diagonal = non-matching (negative!)

   Loss = maximize similarity of N correct pairs
          minimize similarity of N²-N wrong pairs

   L = -(1/N) Σ log[
       exp(sim(iᵢ,tᵢ)/τ) /
       Σⱼ exp(sim(iᵢ,tⱼ)/τ)
   ]
   τ = temperature (learned parameter!)

5. Train on 400M image-text pairs from internet!

ZERO-SHOT CLASSIFICATION:
No fine-tuning needed!

1. Define classes as text: "a photo of a dog"
2. Encode all class texts → text embeddings
3. Encode query image → image embedding
4. Find most similar text embedding
5. That class = prediction!

WHY IT'S REVOLUTIONARY:
→ No labeled data needed for new tasks!
→ "a photo of [class]" → instant classifier
→ Generalizes to unseen categories
→ Enables: image search, zero-shot detection

CLIP VARIANTS:
OpenCLIP: open-source CLIP replications
BLIP:     better caption generation
BLIP-2:   Q-Former bridges vision + LLM
LLaVA:    CLIP + LLaMA = visual chat!
DALL-E:   CLIP guides image generation
"""

# 1. Minimal CLIP implementation
class ImageEncoder(nn.Module):
    """Simplified image encoder"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return self.norm(x)

class TextEncoder(nn.Module):
    """Simplified text encoder"""
    def __init__(self, vocab_size=1000,
                  embed_dim=512, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(
            vocab_size, 256
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )
        self.proj = nn.Linear(256, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pool
        x = self.proj(x)
        return self.norm(x)

class CLIP(nn.Module):
    """Minimal CLIP implementation"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(
            embed_dim
        )
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim
        )
        # Learned temperature!
        self.temperature = nn.Parameter(
            torch.ones([]) * np.log(1/0.07)
        )

    def encode_image(self, images):
        return F.normalize(
            self.image_encoder(images), dim=-1
        )

    def encode_text(self, tokens):
        return F.normalize(
            self.text_encoder(tokens), dim=-1
        )

    def forward(self, images, tokens):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(tokens)

        # Scaled cosine similarity matrix
        logit_scale = self.temperature.exp()
        logits = logit_scale * img_emb @ txt_emb.T
        # logits[i,j] = similarity(image_i, text_j)
        return logits

    def contrastive_loss(self, logits):
        """InfoNCE loss — symmetric!"""
        n = logits.shape[0]
        labels = torch.arange(n)

        # Image → text loss
        loss_i = F.cross_entropy(logits, labels)
        # Text → image loss
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2

# 2. Test CLIP
print("\n=== TESTING CLIP ===")
clip = CLIP(embed_dim=512)

# Fake batch of 8 image-text pairs
images = torch.randn(8, 3, 32, 32)
tokens = torch.randint(0, 1000, (8, 16))

logits = clip(images, tokens)
loss = clip.contrastive_loss(logits)

print(f"Logits shape:   {logits.shape}")
print(f"Contrastive loss: {loss.item():.4f}")
print(f"Temperature:      {clip.temperature.exp().item():.4f}")

# 3. Zero-shot classification demo
print("\n=== ZERO-SHOT CLASSIFICATION ===")
print("""
Real CLIP zero-shot on CIFAR-10:

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# Define classes as text prompts!
class_prompts = [
    "a photo of a airplane",
    "a photo of a car",
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a frog",
    "a photo of a horse",
    "a photo of a ship",
    "a photo of a truck",
]

# Process
inputs = processor(
    text=class_prompts,
    images=image,
    return_tensors="pt",
    padding=True
)
outputs = model(**inputs)

# Zero-shot prediction!
probs = outputs.logits_per_image.softmax(dim=1)
pred_class = probs.argmax().item()
print(f"Predicted: {class_prompts[pred_class]}")
# No fine-tuning! Zero-shot! 🎯

CLIP on CIFAR-10: ~76% zero-shot
ResNet trained on CIFAR-10: ~95%
But CLIP generalizes to ANY classes!
""")

# 4. Similarity search
print("\n=== IMAGE-TEXT SIMILARITY ===")

# Encode images and texts separately
img_embeddings = clip.encode_image(images)
txt_embeddings = clip.encode_text(tokens)

# Cosine similarity
sim_matrix = img_embeddings @ txt_embeddings.T
print(f"Similarity matrix: {sim_matrix.shape}")
print(f"Diagonal (matching pairs):")
print(f"  {sim_matrix.diag().detach().numpy().round(3)}")
print(f"Max similarity: {sim_matrix.max().item():.4f}")

# 5. Databricks + CLIP applications
print("\n=== CLIP AT DATABRICKS SCALE ===")
print("""
Databricks uses CLIP for:
1. Image search on Delta Lake
   → Encode all images → store in Vector DB
   → Query: "find photos of damaged goods"
   → CLIP text encode → nearest neighbor search

2. Mosaic AI Foundation Models
   → CLIP available via API
   → model.predict(images, text_queries)

3. Large-scale CV pipelines
   → Process 1M+ images with Spark
   → CLIP inference distributed across workers
   → Results stored in Delta Lake

4. MLflow tracking CLIP experiments
   → Log text prompts as params
   → Track zero-shot accuracy by prompt
   → Compare "a photo of X" vs "X" prompts

Prompt engineering for CLIP:
"a photo of a {class}" → best for objects
"a satellite photo of {class}" → aerial images
"a medical scan showing {class}" → medical
Context-specific prompts → better accuracy!
""")

# 6. Log to MLflow
mlflow.set_experiment("phase3_clip")
with mlflow.start_run(run_name="CLIP_minimal"):
    mlflow.log_params({
        "model": "CLIP_minimal",
        "embed_dim": 512,
        "loss": "InfoNCE_contrastive",
        "temperature": "learned"
    })
    mlflow.log_metric(
        "contrastive_loss", loss.item()
    )
    print("\nCLIP logged to MLflow!")

print("\n" + "="*60)
print("CLIP + Multimodal — MASTERED! 🎨")
print("="*60)
