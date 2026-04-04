## Behavioral Interview — STAR Format
Situation: Set the context (1-2 sentences)
Task:      What was YOUR responsibility?
Action:    What did YOU specifically do?
           Use "I" not "we"!
Result:    Quantify the outcome!
           "Reduced X by Y%" or "Achieved Z"

## Databricks Values — Map to Stories
Customer obsession → "I prioritized user needs"
Ownership          → "I took end-to-end responsibility"
Humility           → "I acknowledged the mistake"
Bias for action    → "I decided with limited info"
Innovation         → "I found a better approach"

## "Why Databricks" — Key Points
1. Lakehouse solves data+ML gap
2. MLflow — already using it daily
3. Mosaic AI — LLMs at scale excites me
4. Open source DNA — want to contribute
5. Apache Spark — co-creators are here

## LIS — Two Approaches
DP O(n²):     dp[i] = max(dp[j]+1) for j<i
Binary Search: maintain sorted tails array
               bisect_left finds insertion point

## Image Preprocessing Pipeline
BGR → RGB (OpenCV loads as BGR!)
Resize → preserve aspect ratio
Letterbox → add padding to reach target size
Normalize → divide by 255 → [0,1] range
Augment → flip, brightness, blur, rotate
