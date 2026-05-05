# GraphX vs GraphFrames Deep Dive
Date: May 3, 2026

## GraphX (legacy)
API:       Scala/Java only (NO Python!)
Based on:  RDD (not DataFrame)
Algorithms: PageRank, ConnectedComponents,
            TriangleCount, ShortestPaths,
            LabelPropagation, SVD++
Performance: No Catalyst optimization
Status:    Maintenance mode (use GraphFrames!)

## GraphFrames (modern)
API:       Python, Scala, Java ✅
Based on:  DataFrame (Catalyst optimized!)
Algorithms: Same + more
Performance: Catalyst + Tungsten benefits
Status:    Active development

## When to use Graph algorithms at Databricks
Fraud detection: 
  → Transactions as edges, accounts as nodes
  → Find suspicious clusters (ConnectedComponents)
  → Detect money laundering cycles (TriangleCount)

Recommendation systems:
  → Users + items as nodes
  → Interactions as edges
  → PageRank for influential items

Knowledge graphs:
  → Entities as nodes
  → Relationships as edges
  → Shortest paths for entity linking

ML model dependency:
  → Models as nodes (your project!)
  → Dependencies as edges
  → PageRank for critical models

## Interview Answer
"When would you use GraphFrames?"
→ When problem has network/relationship structure
→ When you need distributed graph algorithms
→ Fraud networks, social graphs, knowledge graphs
→ Always prefer GraphFrames over GraphX in Python
→ Checkpoint dir required for connected components!
