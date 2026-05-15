## Ensemble Methods — Three Types

Bagging (parallel):
→ Bootstrap samples + aggregate
→ Reduces VARIANCE
→ Example: Random Forest

Boosting (sequential):
→ Each model corrects previous errors
→ Reduces BIAS
→ Example: XGBoost, AdaBoost, GBM

Stacking:
→ Meta-model on base model predictions
→ Reduces both bias + variance
→ Most powerful, most complex

## AdaBoost vs GBM vs XGBoost

AdaBoost:  reweights samples
           α = (1/2)ln((1-ε)/ε)
GBM:       fits trees to negative gradient
           (residuals of loss function)
XGBoost:   GBM + regularization (L1+L2)
           + second-order optimization
           + column subsampling
           + parallel tree construction

XGBoost wins on:
→ Speed (parallelized)
→ Accuracy (regularized)
→ Missing values (learned direction)

## XGBoost Golden Starting Point
n_estimators=500, learning_rate=0.05
max_depth=6, subsample=0.8
colsample_bytree=0.8
+ early_stopping_rounds=50!

## Week 1 Phase 3 — COMPLETE! 🏆
Day 1: Linear Regression    ✅
Day 2: Logistic Regression  ✅
Day 3: Decision Trees + RF  ✅
Day 4: SVM + kernel trick   ✅
Day 5: Clustering           ✅
Day 6: PCA + t-SNE + UMAP   ✅
Day 7: Ensemble + XGBoost   ✅ today

Week 2 starts tomorrow:
→ Neural network math from scratch!
→ Backpropagation derivation!
→ The foundation of ALL deep learning!
