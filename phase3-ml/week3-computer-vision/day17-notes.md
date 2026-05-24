## Segmentation Task Levels
Classification: 1 label / image
Detection:      boxes / object
Segmentation:   1 label / pixel ← hardest!

Semantic:  same class = same label
Instance:  each object unique
Panoptic:  semantic + instance

## U-Net Architecture
Encoder: DoubleConv → MaxPool (×4)
         saves skip connections at each level!
Bottleneck: deepest features
Decoder: Upsample → Concat(skip) → DoubleConv
Output: 1×1 conv → n_classes channels

Skip connection rule:
encoder_level_N → decoder_level_N
Same spatial resolution → concat channels!
No information lost!

## Loss Functions for Segmentation
CrossEntropy:  standard, good baseline
Dice Loss:     handles class imbalance
               2*TP/(2*TP+FP+FN)
Combined:      CE + Dice (best practice!)
Focal Loss:    focus on hard examples
               for heavily imbalanced classes

## Segmentation Metrics
mIoU:  primary metric
       mean(TP/(TP+FP+FN)) across classes
Dice:  2*intersection/union
       = F1 score for segmentation
Pixel accuracy: misleading for imbalanced!

## Dilated Convolutions (DeepLab)
dilation=r: insert (r-1) zeros between weights
Effective receptive field: k + (k-1)*(r-1)
3×3 conv, r=6: receptive field = 13×13!
→ Large context, same parameters!

## Binary Search Variants
Standard: nums[m]==target
Rotated:  determine sorted half first
Find min: compare nums[m] vs nums[r]
2D matrix: divmod(mid, n) → (row, col)
All share: l<=r, m=(l+r)//2 template!

## CV Week Progress
Day 15: Transfer Learning  ✅
Day 16: YOLOv8            ✅
Day 17: Segmentation      ✅ today
Day 18: ViT from scratch  ← tomorrow!
Day 19: CLIP + multimodal
Day 20: CV + Spark
Day 21: CV Project! 🎯
