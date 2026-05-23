## YOLOv8 Architecture Summary

Backbone (CSPDarknet + C2f):
→ Extracts features at 3 scales (P3/P4/P5)
→ SiLU activation (not ReLU!)
→ C2f = CSP + residual + channel split

Neck (PAN-FPN):
→ FPN: top-down (P5→P3) semantic enrichment
→ PAN: bottom-up (P3→P5) spatial detail
→ Every scale sees ALL other scales!

Head (anchor-free):
→ No pre-defined anchors (YOLOv8 innovation!)
→ DFL for box regression uncertainty
→ Separate cls + box prediction branches

## YOLOv8 Loss Components
Classification: BCE per class
Box regression: CIoU (center + size + aspect)
Distribution:   DFL for anchor-free coords
Total: λ₁*cls + λ₂*box + λ₃*dfl

## Detection Metrics
IoU = Intersection / Union
AP = Area under PR curve
mAP50 = AP at IoU≥0.5
mAP50-95 = mean AP, IoU 0.5:0.95:0.05
Primary metric: mAP50-95!

## NMS Algorithm
1. Sort by confidence (descending)
2. Keep highest confidence box
3. Remove IoU > threshold with kept box
4. Repeat for remaining
conf_threshold=0.25, iou_threshold=0.7

## SiLU Activation (YOLOv8 uses this!)
SiLU(x) = x * sigmoid(x)
Also called Swish activation
Smoother than ReLU, better than GELU for CNN

## Rotated Binary Search Pattern
Determine which half is sorted: nums[l]<=nums[m]
Check if target in sorted half: nums[l]<=t<nums[m]
If yes → search there; else → other half
Key: always one half is sorted!
