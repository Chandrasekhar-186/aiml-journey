# Day 13 — Hypothesis Testing
# Date: March 25, 2026

import numpy as np
from scipy import stats

# 1. t-test — are two models significantly different?
model_a_scores = np.array([88, 92, 85, 90, 87,
                             91, 89, 93, 86, 88])
model_b_scores = np.array([82, 85, 80, 84, 83,
                             86, 81, 85, 82, 84])

t_stat, p_value = stats.ttest_ind(
    model_a_scores, model_b_scores
)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Significant difference (p<0.05): "
      f"{p_value < 0.05}")

# 2. A/B test simulation
np.random.seed(42)
control = np.random.binomial(1, 0.10, 1000)   # 10% CTR
treatment = np.random.binomial(1, 0.12, 1000) # 12% CTR

t_stat, p_value = stats.ttest_ind(
    control, treatment
)
print(f"\nA/B Test Results:")
print(f"Control CTR:   {control.mean():.4f}")
print(f"Treatment CTR: {treatment.mean():.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Launch new version: {p_value < 0.05}")

# 3. Chi-square test
# Are model errors independent of data source?
observed = np.array([[50, 30], [20, 100]])
chi2, p, dof, expected = stats.chi2_contingency(
    observed
)
print(f"\nChi-square: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Independent: {p > 0.05}")
