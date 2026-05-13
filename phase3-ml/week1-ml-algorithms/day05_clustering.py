# Phase 3 Day 5 — Clustering Algorithms
# Date: May 13, 2026
# Unsupervised learning — find structure!

import numpy as np
from sklearn.datasets import (
    make_blobs, make_moons, make_circles
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    davies_bouldin_score
)
import mlflow

print("="*60)
print("Clustering Algorithms — Complete Guide")
print("="*60)

"""
CLUSTERING — OVERVIEW

Unsupervised learning: no labels!
Goal: find natural groupings in data

Main approaches:
1. Centroid-based:  K-Means
2. Density-based:   DBSCAN
3. Hierarchical:    Agglomerative
4. Distribution:    Gaussian Mixture Models

When to use clustering:
→ Customer segmentation
→ Anomaly detection
→ Document grouping
→ Image compression
→ Feature engineering
→ Exploratory data analysis
"""

# ── K-MEANS ──────────────────────────────────────
print("\n=== K-MEANS ALGORITHM ===")
print("""
K-Means Algorithm (Lloyd's algorithm):

OBJECTIVE: minimize Within-Cluster Sum of Squares
WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

ALGORITHM:
1. Initialize K centroids randomly
2. Repeat until convergence:
   a. ASSIGN: each point → nearest centroid
      (Expectation step)
   b. UPDATE: recompute centroids as mean
      of assigned points
      (Maximization step)
3. Converges when assignments stop changing

This is an EM (Expectation-Maximization) algorithm!

GUARANTEES:
→ Always converges (WCSS never increases)
→ NOT guaranteed to find global optimum
→ Solution depends on initialization!

INITIALIZATION MATTERS:
Bad init:  local minimum, poor clusters
K-Means++: smart init, much better results
→ First centroid: random
→ Each next: probability ∝ distance²
→ Spreads centroids apart initially
""")

# Implement K-Means from scratch
class KMeansScratch:
    def __init__(self, k=3, max_iter=300,
                  tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        m = len(X)

        # K-Means++ initialization
        centroids = [X[np.random.randint(m)]]
        for _ in range(self.k - 1):
            # Distance to nearest centroid
            dists = np.array([
                min(np.linalg.norm(x - c)**2
                    for c in centroids)
                for x in X
            ])
            # Sample proportional to distance²
            probs = dists / dists.sum()
            idx = np.random.choice(m, p=probs)
            centroids.append(X[idx])

        self.centroids = np.array(centroids)

        for iteration in range(self.max_iter):
            # E-step: assign to nearest centroid
            distances = np.array([
                [np.linalg.norm(x - c)
                 for c in self.centroids]
                for x in X
            ])
            labels = np.argmin(distances, axis=1)

            # M-step: update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0)
                if sum(labels == k) > 0
                else self.centroids[k]
                for k in range(self.k)
            ])

            # Check convergence
            shift = np.linalg.norm(
                new_centroids - self.centroids
            )
            self.centroids = new_centroids

            if shift < self.tol:
                print(f"Converged at iter {iteration}")
                break

        self.labels = labels
        self.inertia_ = sum(
            np.linalg.norm(
                X[i] - self.centroids[labels[i]]
            )**2
            for i in range(m)
        )
        return self

    def predict(self, X):
        distances = np.array([
            [np.linalg.norm(x - c)
             for c in self.centroids]
            for x in X
        ])
        return np.argmin(distances, axis=1)

# Test on blobs
X_blobs, y_true = make_blobs(
    n_samples=500, centers=4,
    cluster_std=0.8, random_state=42
)

print("\n=== TRAINING K-MEANS ===")
km_scratch = KMeansScratch(k=4)
km_scratch.fit(X_blobs)
sil = silhouette_score(
    X_blobs, km_scratch.labels
)
ari = adjusted_rand_score(
    y_true, km_scratch.labels
)
print(f"Silhouette score: {sil:.4f}")
print(f"Adjusted Rand:    {ari:.4f}")
print(f"Inertia:          "
      f"{km_scratch.inertia_:.2f}")

# sklearn comparison
km_sk = KMeans(n_clusters=4,
                init='k-means++',
                n_init=10, random_state=42)
km_sk.fit(X_blobs)
print(f"sklearn inertia:  {km_sk.inertia_:.2f}")

# Elbow method — find optimal K
print("\n=== ELBOW METHOD ===")
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k,
                 n_init=10, random_state=42)
    km.fit(X_blobs)
    inertias.append(km.inertia_)

print("K → Inertia:")
for k, inertia in zip(K_range, inertias):
    bar = "█" * int(inertia/5000)
    print(f"  K={k}: {inertia:8.1f} {bar}")

print("""
Elbow method: plot K vs Inertia
Look for the "elbow" — diminishing returns
That K = optimal cluster count!

Silhouette score:
→ Measures: how similar a point is to its
  own cluster vs other clusters
→ Range: [-1, 1]
→ 1: perfect clustering
→ 0: overlapping clusters
→ -1: wrong cluster assignment
""")

# ── DBSCAN ───────────────────────────────────────
print("\n=== DBSCAN ALGORITHM ===")
print("""
DBSCAN = Density-Based Spatial Clustering
         of Applications with Noise

KEY INSIGHT: clusters are dense regions
             separated by sparse regions!

Parameters:
  ε (eps):           neighborhood radius
  min_samples:       min points to form core

Point types:
  Core point:    ≥ min_samples in ε-neighborhood
  Border point:  < min_samples but near core
  Noise point:   not reachable from any core

ALGORITHM:
1. For each unvisited point p:
   a. Find all ε-neighbors
   b. If ≥ min_samples → core point
      → expand cluster recursively
   c. If < min_samples → noise (for now)
2. Border points join nearest core cluster

ADVANTAGES over K-Means:
→ Doesn't require K!
→ Finds arbitrary shapes (moons, rings)
→ Handles noise/outliers natively (-1 label)
→ No random initialization!

DISADVANTAGES:
→ Struggles with varying density
→ High-dimensional data (curse of dim.)
→ ε sensitive — requires tuning
""")

# DBSCAN on non-convex shapes
X_moons, _ = make_moons(
    n_samples=400, noise=0.1, random_state=42
)
scaler = StandardScaler()
X_moons_s = scaler.fit_transform(X_moons)

# K-Means fails on moons
km_moons = KMeans(n_clusters=2, random_state=42)
km_labels = km_moons.fit_predict(X_moons_s)
km_sil = silhouette_score(X_moons_s, km_labels)

# DBSCAN succeeds!
dbscan = DBSCAN(eps=0.3, min_samples=5)
db_labels = dbscan.fit_predict(X_moons_s)
n_clusters = len(set(db_labels)) - \
    (1 if -1 in db_labels else 0)
n_noise = sum(db_labels == -1)

print(f"K-Means silhouette: {km_sil:.4f} ← bad!")
if n_clusters > 1:
    db_sil = silhouette_score(
        X_moons_s[db_labels != -1],
        db_labels[db_labels != -1]
    )
    print(f"DBSCAN silhouette:  {db_sil:.4f} ← good!")
print(f"DBSCAN clusters:    {n_clusters}")
print(f"DBSCAN noise pts:   {n_noise}")

# ── HIERARCHICAL ────────────────────────────────
print("\n=== HIERARCHICAL CLUSTERING ===")
print("""
Agglomerative (bottom-up):
1. Start: each point = own cluster
2. Merge: two closest clusters at each step
3. Stop: when k clusters remain

Linkage methods (how to measure distance):
  single:   distance between closest points
            → creates elongated clusters
  complete: distance between farthest points
            → creates compact clusters
  average:  average pairwise distance
            → balanced
  ward:     minimize total within-cluster variance
            → usually best choice!

Dendrogram:
→ Tree showing merge order
→ Cut at any height → different k!
→ Horizontal line at height h → clusters
""")

agg = AgglomerativeClustering(
    n_clusters=4, linkage='ward'
)
agg_labels = agg.fit_predict(X_blobs)
agg_sil = silhouette_score(X_blobs, agg_labels)
agg_ari = adjusted_rand_score(y_true, agg_labels)
print(f"Hierarchical silhouette: {agg_sil:.4f}")
print(f"Hierarchical ARI:        {agg_ari:.4f}")

# ── ALGORITHM COMPARISON ────────────────────────
print("\n=== ALGORITHM COMPARISON ===")
print("""
               K-Means  DBSCAN   Hierarchical
Need K:        YES      NO       YES
Arbitrary shape:NO      YES      Partial
Noise handling: NO      YES      NO
Scalability:   FAST     Medium   SLOW
Global optim:  NO       YES      YES
Use when:      spherical non-convex small data
""")

# Log to MLflow
mlflow.set_experiment("phase3_clustering")
with mlflow.start_run(run_name="clustering_comparison"):
    mlflow.log_param("dataset", "blobs_4_centers")
    mlflow.log_metric("kmeans_silhouette", sil)
    mlflow.log_metric("kmeans_ari", ari)
    mlflow.log_metric("hierarchical_sil",
                       agg_sil)
    mlflow.log_metric("hierarchical_ari",
                       agg_ari)
    mlflow.log_metric("dbscan_clusters",
                       n_clusters)
    print("\nClustering comparison logged!")

print("\n" + "="*60)
print("Clustering Algorithms — MASTERED! 🎯")
print("="*60)
