## CV at Spark Scale

Pandas UDF for inference:
→ Load model ONCE per executor (global var)
→ Process batch of images per call
→ Returns predictions/embeddings
→ Scales linearly with workers!

Delta Lake for images:
→ Store as binary (small) or path (large)
→ ACID writes during batch processing
→ Time travel for dataset versioning!

Horovod: distributed PyTorch on Spark
→ Each worker trains on data partition
→ AllReduce for gradient synchronization
→ Same API as single-machine PyTorch

Mosaic AI Vector Search:
→ Sync from Delta Lake automatically
→ ANN (approximate nearest neighbor)
→ Query with vector → find similar items

## IP Firewall — CIDR Math
ip_to_int: 4 octets → 32-bit integer
           (a<<24)|(b<<16)|(c<<8)|d

mask for /n:
    ((1<<32)-1) - ((1<<(32-n))-1)
    = 11...1 (n ones) 00...0 (32-n zeros)

Match: (ip & mask) == (network & mask)

## Matrix Rotation Pattern
90° clockwise = transpose + reverse rows
90° counter-clockwise = transpose + reverse cols
180° = reverse all rows + reverse each row

## Spiral Matrix Template
top, bottom, left, right = boundaries
Shrink after each direction:
→ left→right: top++
→ top→bottom: right--
→ right→left: bottom-- (if top≤bottom)
→ bottom→top: left++ (if left≤right)

## CV Week Complete Tomorrow!
Day 15: Transfer Learning ✅
Day 16: YOLOv8           ✅
Day 17: U-Net            ✅
Day 18: ViT              ✅
Day 19: CLIP             ✅
Day 20: CV+Spark         ✅ today
Day 21: CV PROJECT!      ← TOMORROW! 🎯
