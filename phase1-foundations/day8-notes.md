## Probability — ML Connections

Bayes Theorem → Naive Bayes classifier
Normal dist   → assumption in Linear Regression
Binomial dist → logistic regression output
Poisson dist  → event counting models
CLT           → why we can use normal approximations

## Key Probability Rules
P(A and B) = P(A) * P(B)  [if independent]
P(A or B)  = P(A) + P(B) - P(A and B)
P(A|B)     = P(A and B) / P(B)  [conditional]

## MLflow — 4 Key Concepts
1. Experiment = collection of related runs
2. Run        = single model training attempt
3. Parameters = inputs (hyperparameters)
4. Metrics    = outputs (accuracy, loss)

## Floyd's Cycle Detection
Slow pointer: moves 1 step
Fast pointer: moves 2 steps
If cycle exists → they will ALWAYS meet
If no cycle   → fast reaches None first
Space: O(1) — beats HashSet approach O(n)

## Linked List — Dummy Node Pattern
Always create dummy = ListNode(0)
Return dummy.next as final answer
Eliminates null pointer edge cases!
