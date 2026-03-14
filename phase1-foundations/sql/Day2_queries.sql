-- Day 02 — SQL Foundations
-- Date: March 14, 2026

-- 1. Filter with WHERE
SELECT name, accuracy 
FROM ml_models 
WHERE accuracy > 90 
ORDER BY accuracy DESC;

-- 2. COUNT with GROUP BY
SELECT model_type, COUNT(*) as total 
FROM ml_models 
GROUP BY model_type;

-- 3. JOIN two tables
SELECT m.name, d.dataset_name 
FROM ml_models m 
JOIN datasets d ON m.dataset_id = d.id;
