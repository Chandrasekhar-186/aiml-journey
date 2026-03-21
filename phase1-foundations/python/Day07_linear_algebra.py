# Day 07 — Linear Algebra for ML
# Date: March 19, 2026

import numpy as np

# 1. Vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Addition:    {v1 + v2}")
print(f"Dot product: {np.dot(v1, v2)}")
print(f"Magnitude:   {np.linalg.norm(v1):.3f}")

# 2. Matrices
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
B = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"A @ B shape: {(A @ B).shape}")   # 3x3
print(f"Matrix multiply:\n{A @ B}")

# 3. Transpose
print(f"Transpose of A:\n{A.T}")

# 4. Eigenvalues (used in PCA!)
square = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(square)
print(f"Eigenvalues:  {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# 5. Real ML use case — cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

model1_embedding = np.array([0.8, 0.2, 0.5])
model2_embedding = np.array([0.9, 0.1, 0.4])
sim = cosine_similarity(model1_embedding,
                         model2_embedding)
print(f"Cosine similarity: {sim:.4f}")
