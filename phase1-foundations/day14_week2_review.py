# Day 14 — Week 2 Self Test
# Date: March 26, 2026
# Write ALL of these from memory — no cheating!

# ── TEST 1: Linear Algebra ──────────────────────
import numpy as np

# Q: Compute cosine similarity between two vectors
def cosine_similarity(a, b):
    dot   = np.dot(a, b)
    norm  = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm          # range [-1, 1]

# Q: What are eigenvalues used for in ML?
# Answer in comment:
'''Eigenvalues measure how much variance a direction (eigenvector) captures in your data.

PCA: you decompose the covariance matrix — the top-k eigenvectors become your principal components, and the eigenvalues tell you the explained variance ratio for each.

Other uses: spectral clustering (graph Laplacian eigenvalues), PageRank, and checking whether a Hessian's eigenvalues are all positive (confirming a loss minimum).'''

# ── TEST 2: Probability ─────────────────────────
from scipy import stats

# Q: Write Bayes theorem for spam classification
# P(Spam|Word) = ?
# Answer in comment:
# P(Spam | Word) = P(Word | Spam) × P(Spam)
#                  ─────────────────────────────
#                          P(Word)
#
# Naive Bayes extends this to multiple words:
# P(Spam | w1…wn) ∝ P(Spam) × ∏ P(wi | Spam)

# Q: What does p-value < 0.05 mean?
# Answer in comment:
'''If the null hypothesis were true, you'd see a result this extreme (or more) less than 5% of the time by random chance.

It does not mean "there's a 95% chance the result is real." It's the probability of the data given H₀, not the probability of H₀ given the data. A small p-value is evidence against H₀, not proof of the alternative..'''

# ── TEST 3: PyTorch ─────────────────────────────
import torch
import torch.nn as nn

# Q: Write a 3-layer neural network from scratch
class ThreeLayerNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# Q: Write the training loop (5 steps)
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:

        # 1. Zero the gradients (prevent accumulation)
        optimizer.zero_grad()

        # 2. Forward pass
        preds = model(X_batch)

        # 3. Compute loss
        loss = criterion(preds, y_batch)

        # 4. Backward pass (compute gradients)
        loss.backward()

        # 5. Update weights
        optimizer.step()

# ── TEST 4: MLflow ──────────────────────────────
import mlflow

# Q: Log params, metrics, and model in one run
# YOUR CODE HERE
with mlflow.start_run():

    # Log hyperparameters (single values, logged once)
    mlflow.log_params({"lr": 0.001, "epochs": 20, "batch_size": 64})

    # Log metrics (can be called per step for curves)
    mlflow.log_metric("accuracy", val_acc, step=epoch)
    mlflow.log_metric("loss",     val_loss, step=epoch)

    # Log the trained model (auto-detects framework)
    mlflow.pytorch.log_model(model, artifact_path="model")

    # Optional: log arbitrary files (plots, configs)
    mlflow.log_artifact("confusion_matrix.png")

# ── TEST 5: SQL Window Functions ────────────────
# Q: Write ROW_NUMBER + RANK + LAG in one query
# YOUR SQL HERE
SELECT
    employee_id,
    department,
    salary,

    -- ROW_NUMBER: unique sequential number, no ties
    ROW_NUMBER() OVER (PARTITION BY department
                        ORDER BY salary DESC)  AS row_num,

    -- RANK: same rank for ties, gaps after ties
    RANK()       OVER (PARTITION BY department
                        ORDER BY salary DESC)  AS rnk,

    -- LAG: previous row's value (NULL for first row)
    LAG(salary, 1) OVER (PARTITION BY department
                          ORDER BY salary DESC) AS prev_salary

FROM employees;
                      
# ── TEST 6: Spark ───────────────────────────────
# Q: What is lazy evaluation?
''' Spark builds a logical execution plan (DAG) when you define transformations (map, filter, join) but does not execute anything yet. Computation only fires when you call an action (collect, count, write). This lets Spark's Catalyst optimizer re-order, fuse, and prune operations before touching data.'''
# Q: When to use broadcast join?
''' When one table is small enough to fit in each executor's memory (rule of thumb: < 10 MB, configurable via spark.sql.autoBroadcastJoinThreshold). Spark sends the small table to every node, eliminating the expensive shuffle of the large table. Use it when joining a large fact table with a small dimension/lookup table.
'''
# Q: Difference between cache() and persist()?
# Answers in comments...
# cache() — shortcut, always uses MEMORY_AND_DISK
df.cache()

# persist() — you choose the StorageLevel
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_ONLY)        # fastest, OOM risk
df.persist(StorageLevel.MEMORY_AND_DISK)    # spills to disk (= cache default)
df.persist(StorageLevel.DISK_ONLY)          # for very large DFs
df.persist(StorageLevel.MEMORY_ONLY_2)      # 2× replicated

# Always unpersist() when done to free memory
df.unpersist()
