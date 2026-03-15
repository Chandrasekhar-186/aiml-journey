-- Day 03 — SQL Aggregations
-- Date: March 15, 2026

-- 1. Basic aggregations
SELECT 
    COUNT(*) as total_models,
    AVG(accuracy) as avg_accuracy,
    MAX(accuracy) as best_accuracy,
    MIN(accuracy) as worst_accuracy
FROM ml_models;

-- 2. GROUP BY — performance per type
SELECT 
    model_type,
    COUNT(*) as count,
    ROUND(AVG(accuracy), 2) as avg_acc
FROM ml_models
GROUP BY model_type
ORDER BY avg_acc DESC;

-- 3. HAVING — filter after grouping
SELECT 
    model_type,
    AVG(accuracy) as avg_acc
FROM ml_models
GROUP BY model_type
HAVING AVG(accuracy) > 85;

-- 4. Subquery
SELECT name, accuracy
FROM ml_models
WHERE accuracy > (
    SELECT AVG(accuracy) FROM ml_models
);
