## Transfer Learning Decision Tree

Dataset size × domain similarity:

Small + similar:   Feature extraction
                   Freeze all, train head only

Small + different: Fine-tune top layers
                   Lower lr for backbone

Large + similar:   Fine-tune all layers
                   Standard lr

Large + different: Train from scratch
                   OR fine-tune all with tiny lr

## Differential Learning Rates
optimizer = AdamW([
  {'params': backbone_params, 'lr': 1e-4},
  {'params': head_params,     'lr': 1e-3}
])
→ Head learns fast (new task!)
→ Backbone learns slow (preserve features!)

## ImageNet Normalization (MEMORIZE!)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
Always use these for pretrained models!

## EfficientNet vs ResNet
EfficientNet-B0: 5.3M params, 77.1% top-1
ResNet-50:       25M params,  76.0% top-1
5× fewer params, BETTER accuracy!
Use EfficientNet for production!

## Merge Sort on Linked List
Find middle (slow/fast)
Split into two halves
Recursively sort each
Merge sorted halves
O(n log n) time, O(log n) space (recursion)

## CV Week Schedule
Day 15: Transfer Learning    ✅ today
Day 16: YOLOv8 internals    ← tomorrow!
Day 17: Segmentation (U-Net)
Day 18: ViT from scratch
Day 19: CLIP + multimodal
Day 20: CV + Spark at scale
Day 21: CV Project! 🎯
