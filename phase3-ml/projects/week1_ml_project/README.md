# Week 1 ML Project — Model Comparison Framework
Date: May 15, 2026

## Problem
Compare all Week 1 algorithms on a real
classification task with proper evaluation.

## Dataset
Use: Scikit-learn breast cancer dataset
     (569 samples, 30 features, binary)
     OR custom tabular dataset

## Algorithms to compare
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- SVM (RBF kernel)
- Gradient Boosting
- XGBoost
- Stacking ensemble

## Evaluation Framework
- 5-fold cross-validation
- Metrics: Accuracy, F1, ROC-AUC
- Feature importance comparison
- Training time comparison
- MLflow experiment tracking

## Key Questions to Answer
1. Which algorithm wins overall?
2. Which is best given limited compute?
3. Does stacking outperform best single model?
4. Which features are most important?
5. How does PCA preprocessing affect results?

## Files
week1_benchmark.py  ← main comparison script
week1_results.md    ← findings + analysis
