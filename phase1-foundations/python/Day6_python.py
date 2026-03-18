# Day 06 — Statistics Foundations
# Date: March 18, 2026

import numpy as np
import pandas as pd

# Sample ML model accuracies
accuracies = np.array([88, 92, 95, 78, 91, 
                        86, 93, 89, 94, 87])

# 1. Central tendency
print(f"Mean:   {np.mean(accuracies):.2f}")
print(f"Median: {np.median(accuracies):.2f}")
print(f"Mode:   {pd.Series(accuracies).mode()[0]}")

# 2. Spread
print(f"Variance: {np.var(accuracies):.2f}")
print(f"Std Dev:  {np.std(accuracies):.2f}")
print(f"Range:    {np.ptp(accuracies)}")

# 3. Normal distribution
mean, std = 90, 5
samples = np.random.normal(mean, std, 1000)
print(f"68% of models score between: "
      f"{mean-std:.1f} and {mean+std:.1f}")
print(f"95% of models score between: "
      f"{mean-2*std:.1f} and {mean+2*std:.1f}")

# 4. Correlation
train_times = np.array([1.2, 2.1, 5.4, 
                         0.8, 3.2, 1.5, 
                         2.8, 1.8, 4.1, 1.3])
correlation = np.corrcoef(accuracies, train_times)[0,1]
print(f"Accuracy vs Train Time correlation: "
      f"{correlation:.3f}")
print("Interpretation: " + 
      ("Positive" if correlation > 0 else "Negative") +
      " correlation")
