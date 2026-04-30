## Cert Final Checklist — 15 Days Out

Must score 90%+ on:
□ Transformation vs Action (100% critical)
□ Window functions syntax
□ Join types (especially semi/anti)
□ Delta Lake operations
□ Streaming output modes + triggers
□ Performance configs (defaults!)
□ RDD key-value operations

## Biggest Traps Recap
cache() = lazy transformation (NOT action!)
coalesce(0) = INVALID (min 1!)
append mode + aggregation = INVALID
no checkpoint = no fault tolerance
past VACUUM = no time travel!
AQE triggers = at shuffle boundaries

## Project README Formula
Problem → Architecture diagram → Results
Tech stack table → Quick start → Key decisions
Skills demonstrated → Why this matters

## String Chain DP Pattern
Sort by length (prerequisite ordering!)
For each word: try removing each char
If predecessor exists: extend its chain
dp[word] = max chain ending at word

## Phase 2 Final Push (15 days)
Day 46-48: Cert final prep
Day 49-52: Advanced streaming + project polish
Day 53-57: Mock interviews + gap filling
Day 58-60: Final cert revision
Day 61:    TAKE THE EXAM! 🎯
