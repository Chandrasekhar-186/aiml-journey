## Top 10 Cert Traps — Answer Instantly

1. cache() → LAZY (transformation!)
2. df.schema → NO parentheses (attribute!)
3. coalesce(10) on 3 parts → STAYS 3
4. append + aggregation → INVALID!
5. persist() no args → MEMORY_ONLY (= cache())
6. Delta checkpoint → every 10 commits
7. VACUUM default → 7 days retention
8. broadcast threshold → 10MB default
9. semi join → LEFT columns ONLY
10. AQE triggers → at shuffle boundaries

## The 2-Point Gap Closers
These 7 questions likely separate 85% from 90%:
→ coalesce cannot increase partitions
→ cache() is lazy not eager
→ df.schema has no parentheses
→ append mode invalid with aggregation
→ persist() with no args = MEMORY_ONLY
→ semi join returns left cols only
→ VACUUM removes past time-travel ability

## Greedy Pattern — Wiggle
Track last direction (up/down)
When direction changes → extend sequence
Never go back → greedy optimal!

## Greedy Sort + Insert Pattern
Sort by primary desc, secondary asc
Insert each element at its index position
→ Works because taller people processed first
→ Shorter people don't affect taller positions

## Exam Day Checklist (12 days!)
□ Read each question TWICE
□ Watch for "which is NOT" questions
□ Eliminate obviously wrong answers
□ For API questions: think exactly
□ For config: remember the defaults
□ Time check: 2 min per question
□ Flag uncertain → return at end
