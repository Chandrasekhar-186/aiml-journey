## SVM Core Math

Objective: maximize margin = 2/||w||
→ Minimize (1/2)||w||²
→ Subject to: yᵢ(w·xᵢ+b) ≥ 1

Soft margin adds slack ξᵢ:
→ Minimize (1/2)||w||² + C*Σξᵢ
→ C = tradeoff: margin vs violations

## Kernel Trick Intuition
Don't compute φ(x) explicitly!
Compute K(x,z) = φ(x)·φ(z) directly.

RBF: K(x,z) = exp(-γ||x-z||²)
→ Similar points: K≈1
→ Different points: K≈0
→ Infinite-dimensional feature space!

## C vs γ (RBF)
C large:   narrow margin, risk overfit
C small:   wide margin, risk underfit
γ large:   tight local fit, risk overfit
γ small:   smooth boundary, risk underfit

## Always scale for SVM!
SVM uses distances → scale-sensitive
StandardScaler before SVM = mandatory

## BST Properties
Inorder traversal = sorted ascending order
LCA in BST: split point = ancestor
            both left = go left
            both right = go right
Kth smallest = kth inorder element

## Week 1 Progress
Day 1: Linear Regression   ✅ GD + Normal Eq
Day 2: Logistic Regression ✅ sigmoid + metrics
Day 3: Decision Trees + RF ✅ Gini + bagging
Day 4: SVM + kernel trick  ✅ today
Day 5: Clustering (tomorrow)
Day 6: PCA + dim reduction
Day 7: Week 1 review + project
