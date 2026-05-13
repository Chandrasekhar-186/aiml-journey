## K-Means Core Math
Minimize WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

E-step: assign each point to nearest centroid
M-step: update centroid = mean of assignments
Repeat until convergence (WCSS never increases)

K-Means++: spread initial centroids apart
→ prob ∝ distance² from existing centroids
→ Much better final solution!

## Elbow Method
Plot K vs WCSS/inertia
Elbow = diminishing returns point = optimal K

## Silhouette Score
s = (b - a) / max(a, b)
a = avg distance to same cluster
b = avg distance to nearest other cluster
s = 1: perfect | 0: overlap | -1: wrong

## DBSCAN Parameters
eps:         neighborhood radius
min_samples: core point threshold
→ Core:   ≥ min_samples in ε-ball
→ Border: < min_samples but near core
→ Noise:  label = -1

Tuning tips:
eps: use k-distance graph elbow
min_samples: start with 2*dimensions

## Algorithm Selection Guide
Spherical clusters, know K: K-Means
Arbitrary shape, unknown K: DBSCAN
Small data, hierarchical view: Agglomerative
Probabilistic assignment: GMM

## DFS Flood Fill = DBSCAN
DFS from seed → expand to neighbors
DBSCAN from core → expand to density-reachable
Both: recursive neighborhood exploration!
Both: mark visited to avoid cycles!
