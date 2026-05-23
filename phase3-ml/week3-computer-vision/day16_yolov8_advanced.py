# Phase 3 CV Day 2 — YOLOv8 Advanced
# Date: May 23, 2026
# From basic runs to production depth!

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import mlflow
import time

print("="*60)
print("YOLOv8 — Production Depth Architecture")
print("="*60)

"""
YOLOV8 ARCHITECTURE — COMPLETE BREAKDOWN

YOLOv8 = You Only Look Once (version 8)
Developed by Ultralytics (2023)

Three main components:
1. BACKBONE:   feature extraction (CSPDarknet)
2. NECK:       feature aggregation (PAN-FPN)
3. HEAD:       detection output (anchor-free!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BACKBONE — CSPDarknet + C2f Modules

CSP = Cross Stage Partial Network
Key idea: split feature maps → process → merge
→ Better gradient flow (like ResNet!)
→ Less computation than standard conv

C2f Module (Cross Stage Partial with 2 convolutions):
Input → split into two branches:
  Branch 1: multiple Bottleneck blocks
  Branch 2: direct connection (skip!)
→ Concat all → 1×1 conv

Bottleneck block:
  Conv 3×3 → Conv 1×1 → residual add
  (like ResNet bottleneck but lighter!)

Backbone stages produce feature maps at:
  P3: stride=8  (1/8 of input, fine details)
  P4: stride=16 (1/16, medium objects)
  P5: stride=32 (1/32, large objects)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. NECK — PAN-FPN (Path Aggregation Network)

Problem: different objects need different scales!
→ Small objects: need high-res (P3)
→ Large objects: need semantic context (P5)

FPN (Feature Pyramid Network):
→ Top-down pathway: upsample P5→P4→P3
→ Gives P3 semantic richness of P5!

PAN (Path Aggregation Network):
→ Bottom-up pathway: P3→P4→P5
→ Gives P5 spatial detail of P3!

YOLOv8 neck = FPN + PAN bidirectional fusion!
→ Every scale gets info from ALL other scales!
→ Detects small AND large objects well!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. HEAD — Anchor-Free Detection

YOLOv7 and earlier: anchor-based
→ Pre-define anchor boxes (bias toward dataset)
→ Predict offsets from anchors
→ Need anchor tuning per dataset

YOLOv8: ANCHOR-FREE! (revolutionary!)
→ Directly predict (x, y, w, h) per cell
→ No anchors to tune!
→ Distribution Focal Loss (DFL) for box reg

Output per scale:
  For each grid cell: predict
  → class probabilities (80 classes COCO)
  → bounding box (x, y, w, h via DFL)

DFL (Distribution Focal Loss):
Instead of predicting scalar (x,y,w,h):
→ Predict probability distribution over values
→ More flexible, handles uncertainty better!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOSS FUNCTIONS:

1. Classification loss: BCE (binary cross-entropy)
   For each class independently

2. Box regression loss: CIoU
   CIoU = IoU - ρ²/c² - αv
   IoU:   intersection over union
   ρ²/c²: penalize center distance
   αv:    penalize aspect ratio difference
   Better than MSE for box regression!

3. Distribution Focal Loss (DFL):
   For anchor-free box coords
   Cross-entropy on distribution

Total loss = λ₁*cls + λ₂*box + λ₃*dfl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NMS (Non-Maximum Suppression):

Problem: multiple boxes for same object!
Solution: keep highest confidence, remove overlapping

Algorithm:
1. Sort boxes by confidence score
2. Keep highest confidence box
3. Remove boxes with IoU > threshold (e.g. 0.5)
   with kept box (overlapping = same object!)
4. Repeat for remaining boxes

YOLOv8 uses: NMS with IoU threshold = 0.7
             Confidence threshold = 0.25
"""

# 1. Build C2f-like module
class Bottleneck(nn.Module):
    """YOLOv8 bottleneck block"""
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = nn.Conv2d(c, c, 3,
                              padding=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.cv2 = nn.Conv2d(c, c, 3,
                              padding=1,
                              bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.shortcut = shortcut

    def forward(self, x):
        out = F.silu(  # SiLU not ReLU in YOLO!
            self.bn1(self.cv1(x))
        )
        out = self.bn2(self.cv2(out))
        return x + out if self.shortcut else out

class C2f(nn.Module):
    """Cross Stage Partial with 2 convolutions"""
    def __init__(self, c_in, c_out,
                  n_bottlenecks=1):
        super().__init__()
        c_hidden = c_out // 2

        self.cv1 = nn.Conv2d(
            c_in, c_out, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(c_out)

        self.bottlenecks = nn.ModuleList([
            Bottleneck(c_hidden)
            for _ in range(n_bottlenecks)
        ])

        # Final conv: merge all branches
        self.cv2 = nn.Conv2d(
            c_hidden * (n_bottlenecks + 2),
            c_out, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        y = F.silu(self.bn1(self.cv1(x)))
        # Split into two halves
        y1, y2 = y.chunk(2, dim=1)

        # Process through bottlenecks
        outputs = [y1, y2]
        for bottleneck in self.bottlenecks:
            y2 = bottleneck(y2)
            outputs.append(y2)

        return F.silu(
            self.bn2(
                self.cv2(torch.cat(outputs,
                                    dim=1))
            )
        )

print("\nC2f module built!")

# 2. Use Ultralytics YOLOv8
print("\n=== ULTRALYTICS YOLOV8 ===")

# Load pretrained YOLOv8n (nano)
model = YOLO('yolov8n.pt')

print(f"Model type: {type(model)}")
print(f"Task: {model.task}")

# 3. Inference on sample image
print("\n=== INFERENCE DEMO ===")

# Run inference (works with URL or local file)
results = model.predict(
    source='https://ultralytics.com/images/bus.jpg',
    conf=0.25,    # confidence threshold
    iou=0.7,      # NMS IoU threshold
    save=False,
    verbose=False
)

for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} objects:")
    if len(boxes) > 0:
        for box in boxes[:5]:  # show first 5
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            name = model.names[cls]
            print(f"  {name}: conf={conf:.3f} "
                  f"box={[round(x,1) for x in xyxy]}")

# 4. Model architecture analysis
print("\n=== MODEL ARCHITECTURE ===")
print(f"Number of parameters: "
      f"{sum(p.numel() for p in model.model.parameters()):,}")

# YOLOv8 variants
print("\nYOLOv8 model variants:")
variants = {
    'yolov8n': ('Nano',    '3.2M',  '37.3'),
    'yolov8s': ('Small',   '11.2M', '44.9'),
    'yolov8m': ('Medium',  '25.9M', '50.2'),
    'yolov8l': ('Large',   '43.7M', '52.9'),
    'yolov8x': ('XLarge',  '68.2M', '53.9'),
}
print(f"{'Variant':10} {'Size':8} "
      f"{'Params':10} {'mAP50-95':>10}")
for key, (size, params, mAP) in variants.items():
    print(f"{key:10} {size:8} "
          f"{params:>10} {mAP:>10}")

# 5. Custom training setup
print("\n=== CUSTOM TRAINING SETUP ===")
print("""
# Train on custom dataset:

# 1. Prepare dataset in YOLO format:
# dataset/
#   images/
#     train/  ← training images
#     val/    ← validation images
#   labels/
#     train/  ← .txt annotation files
#     val/    ← .txt annotation files
#   data.yaml ← dataset config

# data.yaml format:
# path: /path/to/dataset
# train: images/train
# val: images/val
# names:
#   0: cat
#   1: dog
#   2: person

# 2. Train:
model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,           # initial lr
    lrf=0.01,           # final lr factor
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,         # mosaic augmentation!
    mixup=0.1,          # mixup augmentation!
    copy_paste=0.1,     # copy-paste aug!
    project='runs/train',
    name='custom_model',
    save=True,
    plots=True           # generates plots!
)

# 3. Evaluate:
metrics = model.val()
print(metrics.box.map)    # mAP50-95
print(metrics.box.map50)  # mAP50

# 4. Export for deployment:
model.export(format='onnx')   # ONNX
model.export(format='tflite') # TensorFlow Lite
model.export(format='coreml') # Apple CoreML
""")

# 6. IoU and mAP explanation
print("\n=== DETECTION METRICS ===")
print("""
IoU (Intersection over Union):
IoU = Area(Intersection) / Area(Union)
IoU = 1.0: perfect overlap
IoU = 0.0: no overlap
Threshold typically 0.5 (IoU ≥ 0.5 = correct!)

Precision:  TP / (TP + FP)
            Of predicted boxes, % correct
Recall:     TP / (TP + FN)
            Of actual objects, % detected

AP (Average Precision):
Area under Precision-Recall curve
AP50: AP at IoU threshold 0.5
AP50-95: mean AP over IoU 0.5:0.95:0.05

mAP: mean AP across all classes
mAP50-95 is the primary COCO metric!
YOLOv8n achieves mAP50-95 = 37.3 on COCO

Comparison:
Higher threshold → stricter evaluation
mAP50-95 > mAP50 as quality metric!
""")

# 7. Log to MLflow
mlflow.set_experiment("phase3_yolov8")
with mlflow.start_run(
        run_name="YOLOv8_analysis"):
    mlflow.log_params({
        "model": "YOLOv8n",
        "task": "detection",
        "architecture": "CSPDarknet+PAN-FPN",
        "anchor_free": True,
        "loss": "CIoU+DFL+BCE"
    })
    mlflow.log_metrics({
        "n_params": 3200000,
        "coco_map50_95": 37.3,
        "coco_map50": 52.6
    })
    print("\nYOLOv8 analysis logged to MLflow!")

print("\n" + "="*60)
print("YOLOv8 Advanced — MASTERED! 🎯")
print("CV Week Day 2 — COMPLETE!")
print("="*60)
