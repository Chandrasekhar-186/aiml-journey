## Transfer Learning — When & Why

From scratch: need millions of images + weeks of GPU
Transfer learning: reuse pretrained weights → 
                   train only final layer → 
                   great results in hours!

Steps:
1. Load pretrained model (ResNet, VGG, BERT)
2. Freeze all layers (requires_grad=False)
3. Replace final layer for your task
4. Train ONLY final layer (fast!)
5. Optional: unfreeze + fine-tune all layers

## Hypothesis Testing — Interview Answer
H0 (null): no difference between models
H1 (alt):  significant difference exists
p-value < 0.05 → reject H0 → difference is real
p-value > 0.05 → fail to reject H0 → no evidence

## A/B Testing at Databricks scale:
Run experiment → collect metrics →
t-test → p-value → ship or rollback

## Sliding Window Template
left = 0
for right in range(len(arr)):
    # Add arr[right] to window
    while window_invalid:
        # Remove arr[left] from window
        left += 1
    # Update answer with current window
    answer = max(answer, right - left + 1)

## Spark Streaming Key Concepts
readStream  → creates streaming DataFrame
writeStream → outputs to sink
trigger     → how often to process
outputMode  → append/complete/update
queryName   → name for monitoring
