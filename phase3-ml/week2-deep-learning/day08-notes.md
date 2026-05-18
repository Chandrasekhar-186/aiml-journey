## Neural Network Core Math

Forward pass per layer l:
Z[l] = W[l] @ A[l-1] + b[l]
A[l] = g(Z[l])

Backward pass:
dZ[L] = A[L] - Y  (sigmoid + cross-entropy)
dW[l] = (1/m) * dZ[l] @ A[l-1].T
db[l] = (1/m) * sum(dZ[l])
dZ[l-1] = W[l].T @ dZ[l] * g'(Z[l-1])

## Activation Functions
Sigmoid: [0,1], vanishing gradient ← avoid!
ReLU:    max(0,z), gradient=1 for z>0 ← use!
Tanh:    [-1,1], zero-centered, slow
Softmax: multiclass output, sums to 1

## Initialization
Zero init: WRONG (symmetry problem!)
He init:   sqrt(2/n_prev) ← for ReLU ✅
Xavier:    sqrt(1/n_prev) ← for tanh

## Vanishing Gradient
Sigmoid chain: 0.25^L → 0 for deep L
Solution: ReLU + He init + BatchNorm
          + residual connections (ResNet)

## Backprop = Post-order DFS
Process children (layers) before parent
Gradients flow backward: output → input
Chain rule = recursive gradient computation

## Week 2 Preview
Day 8: NN from scratch    ✅ today
Day 9: Backprop deep dive + optimizers
Day 10: Regularization + BatchNorm
Day 11: CNN advanced
Day 12: RNN/LSTM/GRU
Day 13: Attention mechanism deep dive
Day 14: Week 2 review + DL project
→ Week 3 CV starts Day 15! 🎯
