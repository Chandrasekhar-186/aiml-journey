## Trie — When to use
Perfect for: prefix matching, autocomplete,
             spell check, word search
Operations: insert O(m), search O(m),
            startsWith O(m)
            where m = word length

Structure: each node has dict of children
           + boolean is_end flag

## Word Break — DP approach
dp[i] = "can s[:i] be broken into words?"
dp[0] = True (empty string)
dp[i] = True if any dp[j]=True AND s[j:i] in dict

## Delta Lake Partitioning Strategy
Partition by: low-cardinality columns
              (model_type, date, region)
NOT by:       high-cardinality columns (id, uuid)

Partition benefits:
→ Spark skips entire partitions (partition pruning)
→ Faster queries with WHERE on partition column
→ Better file organization

## Capstone Architecture Progress
Day 1: Data generation + MLflow logging ✅
Day 2: PySpark ingestion + Delta Lake ✅
Day 3: Meta-model training (tomorrow)
Day 4: RAG system integration
Day 5: Full system + GitHub README
