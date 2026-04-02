## Generators vs Lists — When to use which

LIST: stores all items in memory at once
→ Use when: small data, need random access, 
  reuse multiple times

GENERATOR: produces items one at a time (lazy)
→ Use when: large data, read once, memory matters

## Why Databricks cares:
Spark is essentially a distributed generator —
it processes data lazily, only computing when
you call .show() or .collect()
This is why Spark is memory efficient at scale!

## HAVING vs WHERE:
WHERE  → filters BEFORE grouping (row level)
HAVING → filters AFTER grouping (group level)
```
