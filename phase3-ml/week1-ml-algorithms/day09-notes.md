## Optimizer Math Summary

SGD:        θ -= lr * g
Momentum:   v = β*v + g; θ -= lr*v
AdaGrad:    G += g²; θ -= lr*g/√G
RMSProp:    G = ρ*G + (1-ρ)*g²; θ -= lr*g/√G
Adam:       m=β₁*m+(1-β₁)*g
            v=β₂*v+(1-β₂)*g²
            θ -= lr * m̂/√v̂  (bias corrected)

## Default Choices
Hidden layers: Adam (lr=0.001)
Fine-tuning:   SGD+Momentum (more stable)
Transformers:  Adam + lr schedule (warmup)
Tabular:       Adam or AdamW

## BatchNorm Math
μ = mean(batch)
σ² = var(batch)
x̂ = (x - μ)/√(σ²+ε)
y = γ*x̂ + β  (γ,β learnable!)

Training: use batch stats
Inference: use running stats (EMA)

## LayerNorm vs BatchNorm
BatchNorm: normalize across BATCH dim
           → bad for small batches
           → bad for sequences
LayerNorm: normalize across FEATURE dim
           → works for any batch size
           → used in Transformers!

## Dropout
Training: zero neurons with prob (1-p)
          scale by 1/p (inverted dropout)
Inference: no dropout, no scaling

## LCA Pattern
If root = p or q → return root
left = LCA(left subtree)
right = LCA(right subtree)
Both non-null → root is LCA!
One non-null → propagate that one up

## Week 2 Progress
Day 8: NN from scratch   ✅
Day 9: Optimizers+BN     ✅ today
Day 10: CNN advanced      (tomorrow)
Day 11: RNN/LSTM/GRU
Day 12: Attention deep
Day 13: DL project
Day 14: Week 2 review
→ Week 3 CV: 5 days away! 🎯
