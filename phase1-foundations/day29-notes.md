## Spark DAG vs MapReduce
MapReduce: disk → process → disk (slow!)
Spark DAG: memory → memory → memory → disk
Result: 10-100× faster for iterative ML!

## Spark Optimization Checklist
1. Run explain() — see what Spark plans
2. Check Spark UI — find slow stages
3. Look for data skew — some tasks 10× slower
4. Check partition count — too few or too many?
5. Missing broadcast hint on small tables?
6. Can you cache() a reused DataFrame?
7. Are you selecting only needed columns?
8. Can filters be pushed earlier?

## Catalyst Optimizer 4 Phases
1. Analysis       → resolve names + types
2. Logical opt    → pushdown, pruning, folding
3. Physical plan  → join strategy selection
4. Code gen       → Tungsten JVM bytecode

## Merge K Sorted Lists — Heap Pattern
Push (value, index, node) to min-heap
index breaks ties — Python can't compare nodes!
Pop min → add to result → push node.next

## BST Validation — Bounds Pattern
Pass (min_val, max_val) bounds recursively
Left child:  max_val = parent.val
Right child: min_val = parent.val
Invalid if: node.val <= min OR node.val >= max

## Phase 1 Final Numbers
Days:      29/31
LeetCode:  53 (15E + 32M + 4H)
Projects:  2 complete
Posts:     28 LinkedIn
Streak:    🔥 29 days
Score:     88.75% Superday
