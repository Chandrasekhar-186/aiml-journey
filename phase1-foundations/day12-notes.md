## CNN Architecture — Layer by Layer

Input Image (H x W x C)
    ↓
Conv2d → extracts local features (edges, textures)
    ↓
ReLU  → adds non-linearity (learn complex patterns)
    ↓
MaxPool → reduces spatial size (translation invariant)
    ↓
[Repeat Conv+ReLU+Pool blocks]
    ↓
Flatten → convert 3D feature map to 1D vector
    ↓
Linear (FC) → classify based on features
    ↓
Output (num_classes)

## Key CNN Concepts
Filter/Kernel: small matrix that slides over image
Feature map: output of Conv layer
Stride: how many pixels filter moves each step
Padding: add zeros around image edges
MaxPool: take maximum value in each region

## Spark Optimization — Top 5 Rules
1. Combine filters into single operation
2. Cache DataFrames used multiple times
3. Use broadcast join for small tables (<10MB)
4. Repartition on join/groupBy key
5. Avoid UDFs — use built-in F.functions

## Dynamic Programming Pattern
1. Define dp array (dp[i] = answer for i)
2. Set base case (dp[0] = 0 or 1)
3. Fill array bottom-up using recurrence
4. Return dp[target]

Coin Change recurrence:
dp[amt] = min(dp[amt], dp[amt-coin] + 1)
