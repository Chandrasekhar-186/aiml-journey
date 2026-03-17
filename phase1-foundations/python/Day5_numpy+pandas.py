# Day 05 — NumPy + Pandas
# Date: March 17, 2026

import numpy as np
import pandas as pd

# ===== NUMPY =====
# 1. Array operations
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")
print(f"Std:  {np.std(arr)}")
print(f"Shape: {arr.shape}")

# 2. Matrix operations (critical for ML!)
matrix = np.random.randn(3, 3)
print(f"Matrix:\n{matrix}")
print(f"Transpose:\n{matrix.T}")
print(f"Dot product:\n{matrix @ matrix.T}")

# 3. Broadcasting
accuracies = np.array([88, 92, 95, 78, 91])
normalized = (accuracies - accuracies.mean()) / accuracies.std()
print(f"Normalized: {normalized}")

# ===== PANDAS =====
# 4. Create DataFrame
data = {
    'model': ['RF', 'XGB', 'NN', 'SVM', 'LR'],
    'accuracy': [88, 92, 95, 78, 82],
    'train_time': [1.2, 2.1, 5.4, 0.8, 0.3],
    'framework': ['sklearn', 'xgboost', 
                  'pytorch', 'sklearn', 'sklearn']
}
df = pd.DataFrame(data)

# 5. Essential operations
print(df.describe())               # stats summary
print(df[df['accuracy'] > 85])    # filter
print(df.groupby('framework')['accuracy'].mean())  # groupby
print(df.sort_values('accuracy', ascending=False))  # sort

# 6. Missing data handling
df.loc[2, 'train_time'] = np.nan
print(f"Nulls:\n{df.isnull().sum()}")
df['train_time'].fillna(df['train_time'].mean(), 
                         inplace=True)

# 7. Apply function
df['grade'] = df['accuracy'].apply(
    lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C'
)
print(df)
