-- Day 05 — DDL + DML Operations
-- Date: March 17, 2026

-- 1. CREATE TABLE
CREATE TABLE ml_experiments (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    accuracy FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. INSERT
INSERT INTO ml_experiments (model_name, accuracy)
VALUES 
    ('RandomForest', 92.5),
    ('XGBoost', 94.1),
    ('NeuralNet', 96.3);

-- 3. UPDATE
UPDATE ml_experiments
SET accuracy = 95.0
WHERE model_name = 'RandomForest';

-- 4. DELETE
DELETE FROM ml_experiments
WHERE accuracy < 90;

-- 5. ALTER TABLE
ALTER TABLE ml_experiments
ADD COLUMN framework TEXT DEFAULT 'sklearn';
