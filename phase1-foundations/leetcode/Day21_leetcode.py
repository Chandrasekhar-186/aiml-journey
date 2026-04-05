# Day 21 — LeetCode #684 Redundant Connection
# Difficulty: Medium ⚡
# Approach: Union-Find
# Time: O(n α(n)) ≈ O(n) | Space: O(n)

def findRedundantConnection(edges):
    n = len(edges)
    parent = list(range(n + 1))
    rank = [0] * (n + 1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False  # already connected = cycle!
        # Union by rank
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    for u, v in edges:
        if not union(u, v):
            return [u, v]  # this edge creates cycle!

print(findRedundantConnection(
    [[1,2],[1,3],[2,3]]))    # [2,3]
print(findRedundantConnection(
    [[1,2],[2,3],[3,4],[1,4],[1,5]]))  # [1,4]


# Day 21 — LeetCode #48 Rotate Image
# Difficulty: Medium ⚡
# Approach: Transpose + reverse rows
# Time: O(n²) | Space: O(1)

def rotate(matrix):
    n = len(matrix)

    # Step 1: Transpose (swap across diagonal)
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = (
                matrix[j][i], matrix[i][j]
            )

    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()

    return matrix

matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(rotate(matrix))  # [[7,4,1],[8,5,2],[9,6,3]]
