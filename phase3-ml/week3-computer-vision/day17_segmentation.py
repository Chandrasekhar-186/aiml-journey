# Phase 3 CV Day 3 — Semantic Segmentation
# Date: May 24, 2026
# Every pixel classified — the hardest CV task!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import mlflow

print("="*60)
print("Semantic Segmentation — U-Net + DeepLab")
print("="*60)

"""
SEGMENTATION — OVERVIEW

Three levels of scene understanding:
1. Classification:  one label per IMAGE
2. Detection:       bounding boxes per OBJECT
3. Segmentation:    one label per PIXEL ← hardest!

Segmentation types:
Semantic:   same class = same label (all cars = red)
Instance:   each object = unique label (car1≠car2)
Panoptic:   semantic + instance combined

Applications:
→ Medical imaging (tumor segmentation)
→ Autonomous driving (road/pedestrian/car)
→ Satellite imagery (land use mapping)
→ Industrial inspection (defect detection)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FCN (Fully Convolutional Network, 2015):
First end-to-end segmentation network!

Key insight: replace FC layers with 1×1 convs!
→ FC layer: loses spatial information
→ 1×1 conv: preserves spatial info!
→ Any input size works!

Architecture:
Backbone (VGG) → downsample (1/32)
               → upsample (bilinear)
               → pixel predictions

Problem: coarse predictions (lost detail!)
→ 32× upsample = blurry boundaries

FCN-16s, FCN-8s: use skip connections!
→ Add intermediate feature maps before upsampling
→ Better boundary detail!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

U-NET (2015) — THE MEDICAL IMAGING STANDARD:

Architecture: symmetric encoder-decoder
  Encoder: downsample (like CNN backbone)
  Decoder: upsample (mirror of encoder)
  Skip connections: encoder → decoder at each level!

WHY U-NET DOMINATES MEDICAL IMAGING:
1. Skip connections: preserve fine details!
   → Low-level features (edges, textures)
     directly fed to decoder
   → No information lost during downsampling!

2. Works with small datasets:
   → Data augmentation (elastic deformation!)
   → Trained on 30 images originally!

3. Full resolution output:
   → Pixel-perfect segmentation masks

U-Net shape (for 572×572 input):
Encoder:
  Conv3×3 → ReLU → Conv3×3 → ReLU → MaxPool
  64 → 128 → 256 → 512 → 1024 channels

Bottleneck: 1024 channels, smallest spatial

Decoder:
  ConvTranspose2x2 → Concat(skip) → Conv→Conv
  1024 → 512 → 256 → 128 → 64 channels

Output: 1×1 conv → num_classes channels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEGMENTATION METRICS:

IoU (Intersection over Union) per class:
IoU_c = TP_c / (TP_c + FP_c + FN_c)

mIoU (mean IoU):
mIoU = (1/C) Σ IoU_c
Primary metric for segmentation!

Pixel Accuracy: correct_pixels / total_pixels
(misleading if class imbalance!)

Dice coefficient (used in medical):
Dice = 2*|A∩B| / (|A|+|B|)
= 2*TP / (2*TP + FP + FN)
Equivalent to F1 score for segmentation!
"""

# 1. Double Conv block (U-Net building block)
class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks"""
    def __init__(self, in_ch, out_ch,
                  mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch,
                       3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch,
                       3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 2. Encoder (downsampling) block
class Down(nn.Module):
    """MaxPool then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

# 3. Decoder (upsampling) block
class Up(nn.Module):
    """Upsample then concat skip + DoubleConv"""
    def __init__(self, in_ch, out_ch,
                  bilinear=True):
        super().__init__()
        if bilinear:
            # Bilinear upsample + conv
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
            self.conv = DoubleConv(
                in_ch, out_ch, in_ch // 2
            )
        else:
            # Transposed convolution
            self.up = nn.ConvTranspose2d(
                in_ch, in_ch // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """x1: from decoder, x2: skip from encoder"""
        x1 = self.up(x1)

        # Handle size mismatch (padding)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [
            diff_x//2, diff_x - diff_x//2,
            diff_y//2, diff_y - diff_y//2
        ])

        # Concatenate skip connection!
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 4. Complete U-Net
class UNet(nn.Module):
    def __init__(self, n_channels=3,
                  n_classes=2, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder (expanding path)
        self.up1 = Up(1024, 512 // factor,
                       bilinear)
        self.up2 = Up(512, 256 // factor,
                       bilinear)
        self.up3 = Up(256, 128 // factor,
                       bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output: 1×1 conv
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder: save skip connections!
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        # Decoder: use skip connections!
        x = self.up1(x5, x4)  # concat x4!
        x = self.up2(x, x3)   # concat x3!
        x = self.up3(x, x2)   # concat x2!
        x = self.up4(x, x1)   # concat x1!

        # Output logits (no softmax yet!)
        return self.outc(x)

# 5. Test U-Net
print("\n=== U-NET ARCHITECTURE TEST ===")
model = UNet(n_channels=3, n_classes=10)
x_test = torch.randn(2, 3, 256, 256)
out = model(x_test)
print(f"Input:  {x_test.shape}")
print(f"Output: {out.shape}")
# Should be: (2, 10, 256, 256) — same spatial!

total_params = sum(
    p.numel() for p in model.parameters()
)
print(f"Parameters: {total_params:,}")
print("Output same spatial size as input ✅")

# 6. Segmentation loss functions
print("\n=== SEGMENTATION LOSSES ===")

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) class indices
        pred = F.softmax(pred, dim=1)
        # One-hot encode target
        n_classes = pred.shape[1]
        target_oh = F.one_hot(
            target, n_classes
        ).permute(0, 3, 1, 2).float()

        intersection = (pred * target_oh).sum(
            dim=(2, 3)
        )
        union = pred.sum(dim=(2,3)) + \
                target_oh.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / \
               (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """CE + Dice (best for segmentation!)"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return (self.alpha * self.ce(pred, target) +
                (1-self.alpha) *
                self.dice(pred, target))

# Test losses
criterion = CombinedLoss(alpha=0.5)
pred = torch.randn(2, 10, 256, 256)
target = torch.randint(0, 10, (2, 256, 256))
loss = criterion(pred, target)
print(f"\nCombined loss test: {loss.item():.4f}")

# 7. IoU metric implementation
def compute_iou(pred, target, n_classes):
    """Compute mIoU"""
    pred = pred.argmax(dim=1)  # (B, H, W)
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union == 0:
            continue
        ious.append(
            (intersection / union).item()
        )
    return np.mean(ious) if ious else 0.0

miou = compute_iou(pred, target, 10)
print(f"mIoU (random):      {miou:.4f}")

# 8. DeepLab overview
print("\n=== DEEPLAB ARCHITECTURE ===")
print("""
DeepLab series (Google Brain):
DeepLabv1 (2015): CRF post-processing
DeepLabv2 (2017): ASPP module
DeepLabv3 (2018): Improved ASPP
DeepLabv3+ (2018): Encoder-decoder + ASPP

ASPP (Atrous Spatial Pyramid Pooling):
Key innovation: dilated (atrous) convolutions!

Dilated conv: conv with gaps between weights
→ dilation=1: standard 3×3 (receptive field=3)
→ dilation=6: 3×3 with gaps (receptive field=13!)
→ dilation=12: receptive field=25!
→ dilation=18: receptive field=37!

ASPP: apply 4 parallel dilated convs
→ Captures multi-scale context!
→ No parameters added (same filter size)
→ Just dilation rate changes!

DeepLabv3+ performance:
mIoU=89.0% on Pascal VOC 2012 (SOTA!)

When to use:
U-Net:    medical imaging, small datasets
DeepLab:  general semantic segmentation
YOLOv8:  instance segmentation (masks+boxes)
""")

# 9. Quick training demo
print("\n=== TRAINING DEMO ===")
device = torch.device('cpu')
model = UNet(n_channels=3, n_classes=5)
model = model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4
)
criterion = CombinedLoss()

# Synthetic data
X_demo = torch.randn(4, 3, 128, 128)
y_demo = torch.randint(0, 5, (4, 128, 128))

mlflow.set_experiment("phase3_segmentation")
with mlflow.start_run(
        run_name="UNet_segmentation"):
    mlflow.log_params({
        "model": "U-Net",
        "n_classes": 5,
        "loss": "CE+Dice",
        "optimizer": "Adam",
        "lr": 1e-4
    })

    for step in range(5):
        model.train()
        optimizer.zero_grad()
        out = model(X_demo)
        loss = criterion(out, y_demo)
        loss.backward()
        optimizer.step()

        miou = compute_iou(
            out.detach(), y_demo, 5
        )
        mlflow.log_metrics({
            "loss": loss.item(),
            "miou": miou
        }, step=step)
        print(f"Step {step+1}: "
              f"loss={loss.item():.4f} "
              f"mIoU={miou:.4f}")

    print("\nU-Net training logged to MLflow!")

print("\n" + "="*60)
print("Semantic Segmentation — MASTERED! 🎨")
print("CV Week Day 3 — COMPLETE!")
print("="*60)
