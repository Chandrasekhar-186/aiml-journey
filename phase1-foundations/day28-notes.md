## LinkedIn Optimization — Recruiter Keywords
Databricks recruiters search these exact terms:
"Apache Spark" "MLflow" "Delta Lake" "PySpark"
"LLM" "RAG" "PyTorch" "MLOps" "Computer Vision"

All must appear in:
→ Headline (most weight)
→ About section
→ Skills section (endorsements help)
→ Project descriptions

## Open Source Contribution Strategy
Week 1: Setup + read 10 issues
Week 2: Pick documentation issue → submit PR
Week 3: Follow reviewer feedback → merge
Week 4: Pick code issue → submit PR
Goal: 2 merged PRs before Phase 5 ends

## Tree DP Pattern
Post-order DFS (process children first!)
At each node:
  left_gain  = max(0, dfs(left))
  right_gain = max(0, dfs(right))
  update_global_max(node.val + left + right)
  return node.val + max(left, right)

Key: return SINGLE path to parent
     update GLOBAL max with both paths

## Palindrome Expand Pattern
For each center (n centers for odd,
n-1 centers for even):
  expand outward while s[l] == s[r]
  count each valid palindrome
Time: O(n²) — optimal without Manacher's

## Phase 2 Key Focus Areas
1. Spark internals (DAG, Catalyst, Tungsten)
2. RDD operations (exam requires this!)
3. Delta Lake advanced (liquid clustering)
4. Structured Streaming + Kafka
5. Spark cert exam by end of Phase 2!
