## DL Project — Production Tricks Used

Label smoothing (0.1):
→ Instead of hard 0/1 targets
→ Use 0.05 for wrong, 0.95 for right
→ Prevents overconfident predictions
→ Better calibration + generalization

OneCycleLR scheduler:
→ Warmup → peak lr → cooldown
→ Much faster convergence than fixed lr
→ Single best hyperparameter choice!

Gradient clipping (max_norm=1.0):
→ Caps gradient magnitude
→ Prevents exploding gradients
→ Essential for RNNs, useful for CNNs

AdamW (not Adam):
→ Decoupled weight decay
→ Better regularization than Adam+L2
→ Default for most modern DL

Channel Attention (SE block):
→ Squeeze: global avg pool → vector
→ Excitation: FC → FC → sigmoid
→ Scale: multiply channels by weights
→ "Let network learn which channels matter"

## Floyd's Algorithm
Phase 1: slow+1, fast+2 per step
         meet somewhere in cycle
Phase 2: reset slow to head
         advance both by 1
         meet at cycle START!

Math proof:
F = steps to cycle start
Meeting point is F steps into cycle
→ Reset + advance F steps = cycle start!

## Week 2 DL Complete! 🏆
Day 8:  NN from scratch   ✅
Day 9:  Optimizers+BN+DO  ✅
Day 10: CNN+ResNet        ✅
Day 11: RNN+LSTM+GRU      ✅
Day 12: Attention+Trans   ✅
Day 13: DL Project        ✅ today
Day 14: Week 2 review     ← tomorrow
→ CV WEEK STARTS DAY 15! TOMORROW! 🎨
