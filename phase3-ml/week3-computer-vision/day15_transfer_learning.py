# Phase 3 CV Day 1 — Transfer Learning
# Date: May 22, 2026
# How 90% of real CV systems are built!

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader, Dataset
)
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import mlflow
import mlflow.pytorch
import time
import numpy as np

print("="*60)
print("Transfer Learning — Production CV")
print("="*60)

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)

"""
TRANSFER LEARNING — WHY IT WORKS

ImageNet training: 1.2M images, 1000 classes
→ Model learns universal visual features:
  Layer 1-3:  edges, textures, colors
  Layer 4-6:  shapes, patterns, parts
  Layer 7-9:  object parts, semantics
  Layer 10+:  class-specific features

Key insight: LOWER layers are UNIVERSAL!
→ Edges look the same in dogs vs cars
→ Textures generalize across datasets
→ Only TOP layers are domain-specific!

Transfer learning strategies:

1. FREEZE ALL + new head (feature extraction):
   → Fastest (only train last FC)
   → Best for tiny datasets (<1000 samples)
   → Least flexible

2. FREEZE BACKBONE + train head:
   → Train only classification head
   → Best for small datasets
   → Most common approach!

3. FINE-TUNE TOP LAYERS:
   → Unfreeze last few blocks + head
   → Better for medium datasets
   → Uses lower learning rate!

4. FINE-TUNE ALL (end-to-end):
   → Update all weights (slowly!)
   → Best for large datasets
   → Risk of catastrophic forgetting!

Learning rate rule for fine-tuning:
  New head:      lr = 0.001
  Top blocks:    lr = 0.0001
  Lower blocks:  lr = 0.00001
  (10× smaller per layer group!)
"""

# 1. Load pretrained EfficientNet
print("\n=== EFFICIENTNET-B0 ===")
print("""
EfficientNet compound scaling:
→ Scale depth + width + resolution TOGETHER
→ Based on neural architecture search (NAS)
→ B0 (smallest) to B7 (largest)
→ EfficientNet-B0: 5.3M params, 77.1% top-1
→ ResNet-50:       25M params,  76.0% top-1
→ 5× fewer params, BETTER accuracy!
""")

# Load pretrained EfficientNet-B0
efficientnet = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

# Inspect architecture
print("EfficientNet-B0 structure:")
print(f"  Features: {type(efficientnet.features)}")
print(f"  Classifier: {efficientnet.classifier}")

# 2. Strategy 1: Feature extraction
def build_feature_extractor(
        base_model, num_classes, dropout=0.3):
    """Freeze all backbone, replace classifier"""
    model = base_model

    # Freeze ALL backbone params
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout/2),
        nn.Linear(256, num_classes)
    )
    return model

# 3. Strategy 2: Fine-tuning (recommended!)
def build_finetuned_model(
        num_classes, unfreeze_blocks=3,
        dropout=0.3):
    """Unfreeze top N blocks for fine-tuning"""
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights
                       .IMAGENET1K_V1
    )

    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze top blocks
    blocks = list(model.features.children())
    for block in blocks[-unfreeze_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Replace + unfreeze classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout/2),
        nn.Linear(256, num_classes)
    )

    # Count trainable params
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    total = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Trainable: {trainable:,}/{total:,} "
          f"({100*trainable/total:.1f}%)")
    return model

# 4. Data pipeline for CIFAR-10
# (pretrained models expect 224×224!)
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],  # ImageNet mean!
        [0.229, 0.224, 0.225]   # ImageNet std!
    )
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_set = torchvision.datasets.CIFAR10(
    './data', train=True, download=True,
    transform=transform_train
)
test_set = torchvision.datasets.CIFAR10(
    './data', train=False, download=True,
    transform=transform_test
)

# Use small subset for speed
train_subset = torch.utils.data.Subset(
    train_set, range(2000)
)
test_subset = torch.utils.data.Subset(
    test_set, range(500)
)

train_loader = DataLoader(
    train_subset, batch_size=32,
    shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_subset, batch_size=32,
    shuffle=False, num_workers=2
)

# 5. Build model with differential LR
print("\n=== BUILDING FINETUNED MODEL ===")
model = build_finetuned_model(
    num_classes=10,
    unfreeze_blocks=3,
    dropout=0.3
)
model = model.to(device)

# Differential learning rates!
backbone_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and 'classifier' not in n
]
head_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and 'classifier' in n
]

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': 1e-4},
    {'params': head_params, 'lr': 1e-3}
], weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# 6. Training
def train_eval(model, train_loader,
                test_loader, optimizer,
                scheduler, criterion,
                epochs=10):
    best_acc = 0
    for epoch in range(epochs):
        # Train
        model.train()
        tr_correct = tr_total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            pred = out.argmax(1)
            tr_correct += (pred==y).sum().item()
            tr_total += y.size(0)

        # Eval
        model.eval()
        te_correct = te_total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                te_correct += (pred==y).sum().item()
                te_total += y.size(0)

        tr_acc = tr_correct/tr_total
        te_acc = te_correct/te_total
        scheduler.step()

        if te_acc > best_acc:
            best_acc = te_acc

        if epoch % 2 == 0:
            print(f"Ep {epoch+1:2d}: "
                  f"train={tr_acc:.4f} "
                  f"test={te_acc:.4f}")

    return best_acc

mlflow.set_experiment("phase3_cv_transfer")
with mlflow.start_run(
        run_name="EfficientNet_CIFAR10"):
    mlflow.log_params({
        "model": "EfficientNet-B0",
        "strategy": "fine-tune-top-3",
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "unfreeze_blocks": 3,
        "epochs": 10
    })

    print("\nTraining with differential LRs:")
    best_acc = train_eval(
        model, train_loader, test_loader,
        optimizer, scheduler, criterion,
        epochs=10
    )

    mlflow.log_metric("best_acc", best_acc)
    mlflow.pytorch.log_model(model, "model")
    print(f"\nBest accuracy: {best_acc:.4f}")
    print("Model logged to MLflow registry!")

# 7. Transfer learning decision guide
print("\n=== TRANSFER LEARNING GUIDE ===")
print("""
Dataset size → Strategy:

<500 samples:  Feature extraction only
               Freeze all, train head
               Risk: underfit

500-5K:        Fine-tune top layers
               Unfreeze last 2-3 blocks
               lr_backbone = 1e-4 ✅

5K-50K:        Fine-tune more layers
               Unfreeze last 5-6 blocks
               lr decay per layer group

>50K:          Full fine-tune or train from scratch
               Lower lr throughout
               More augmentation!

Domain similarity → Strategy:
Similar domain:  Fine-tune less (features transfer!)
Different domain: Fine-tune more (features don't!)
Example: ImageNet→Medical: fine-tune ALL!
         ImageNet→CIFAR: fine-tune top layers!
""")

print("\n" + "="*60)
print("Transfer Learning — MASTERED! 🎨")
print("CV Week Day 1 — COMPLETE!")
print("="*60)
