-- Day 04 — Window Functions
-- Date: March 16, 2026
-- THE most important SQL topic for Databricks!

-- 1. ROW_NUMBER — unique rank per partition
SELECT 
    name,
    model_type,
    accuracy,
    ROW_NUMBER() OVER (
        PARTITION BY model_type 
        ORDER BY accuracy DESC
    ) as rank_in_type
FROM ml_models;

-- 2. RANK vs DENSE_RANK
SELECT 
    name,
    accuracy,
    RANK() OVER (ORDER BY accuracy DESC) as rank,
    DENSE_RANK() OVER (ORDER BY accuracy DESC) as dense_rank
FROM ml_models;
-- RANK: 1,2,2,4 (skips 3)
-- DENSE_RANK: 1,2,2,3 (no skip)

-- 3. LAG and LEAD — compare with previous/next row
SELECT 
    name,
    accuracy,
    LAG(accuracy) OVER (ORDER BY created_at) as prev_accuracy,
    LEAD(accuracy) OVER (ORDER BY created_at) as next_accuracy,
    accuracy - LAG(accuracy) OVER 
        (ORDER BY created_at) as improvement
FROM ml_models;

-- 4. Running total
SELECT 
    name,
    accuracy,
    SUM(accuracy) OVER (
        ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total
FROM ml_models;

-- 5. Top 1 model per type (classic interview question!)
SELECT * FROM (
    SELECT 
        name, model_type, accuracy,
        ROW_NUMBER() OVER (
            PARTITION BY model_type 
            ORDER BY accuracy DESC
        ) as rn
    FROM ml_models
) WHERE rn = 1;
