## ML Pipeline — Standard Steps
1. Load & explore data (EDA)
2. Feature engineering
3. Train/test split (stratify for imbalanced!)
4. Scale features (StandardScaler for linear models)
5. Train model
6. Evaluate (accuracy, precision, recall, F1)
7. Log everything to MLflow
8. Compare runs in MLflow UI

## Precision vs Recall — When to use which
Precision = TP / (TP + FP)
→ Use when: False Positives are costly
→ Example: Spam filter (don't block legit emails)

Recall = TP / (TP + FN)
→ Use when: False Negatives are costly
→ Example: Cancer detection (don't miss cases!)

F1 = 2 * (Precision * Recall) / (Precision + Recall)
→ Use when: Both matter equally

## Binary Search — Template
left, right = 0, len(nums) - 1
while left <= right:
    mid = left + (right - left) // 2
    if condition: return mid
    elif go_right: left = mid + 1
    else: right = mid - 1

Key: mid = left + (right-left)//2
Never: mid = (left+right)//2  ← integer overflow!

## Feature Engineering Rules
→ Always fit scaler on TRAIN only
→ Apply same scaler to TEST (no data leakage!)
→ Log transform skewed features
→ Drop highly correlated features (>0.95)
