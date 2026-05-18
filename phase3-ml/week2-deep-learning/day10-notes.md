## Convolution Output Size
H_out = (H_in - K + 2P)/S + 1
K=kernel, P=padding, S=stride

Common configs:
3×3, P=1, S=1: same size (most common!)
3×3, P=1, S=2: half size (downsampling)
1×1, P=0, S=1: channel mixing only

## ResNet Key Insight
Regular: y = F(x)
Residual: y = F(x) + x  ← skip connection!

If identity is optimal: F(x) learns 0
Gradient: ∂L/∂x = ∂L/∂y * (∂F/∂x + 1)
The "+1" = guaranteed gradient flow!

## ResNet Block Rule
Same spatial size: identity shortcut
Different size/channels: 1×1 conv shortcut

## Architecture Evolution
VGG(2014) → ResNet(2015) → DenseNet(2016)
→ EfficientNet(2019) → ConvNeXt(2022)
Each iteration: better accuracy/compute!

## Path Sum Pattern
Prefix sum HashMap on tree = subarray sum!
curr_sum - target in map → path exists
ALWAYS backtrack after DFS:
prefix_count[curr_sum] -= 1
→ Remove state after exploring subtree!

## Week 2 Day 3 of 7
Day 8: NN scratch    ✅
Day 9: Optimizers+BN ✅
Day 10: CNN+ResNet   ✅ today
Day 11: RNN/LSTM     ← tomorrow
Day 12: Attention
Day 13: DL Project
Day 14: Review
→ CV Week starts Day 15: 4 DAYS! 🎯
