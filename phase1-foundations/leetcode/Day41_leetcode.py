# LeetCode #1765 — Map of Highest Peak
# Difficulty: Medium ⚡
# Same pattern as Walls and Gates!
# Multi-source BFS from all water cells (0)
# Time: O(m*n) | Space: O(m*n)

from collections import deque

def highestPeak(isWater):
    m, n = len(isWater), len(isWater[0])
    height = [[-1]*n for _ in range(m)]
    queue = deque()

    # Start BFS from ALL water cells simultaneously
    # Same as Walls and Gates starting from gates!
    for r in range(m):
        for c in range(n):
            if isWater[r][c] == 1:
                height[r][c] = 0
                queue.append((r, c))

    directions = [(0,1),(0,-1),(1,0),(-1,0)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if (0 <= nr < m and
                0 <= nc < n and
                height[nr][nc] == -1):
                height[nr][nc] = height[r][c] + 1
                queue.append((nr, nc))

    return height

# Test cases
print(highestPeak([[0,1],[0,0]]))
# [[1,0],[2,1]]
print(highestPeak([[0,0,1],[1,0,0],[0,0,0]]))
# [[1,1,0],[0,1,1],[1,2,2]]


# LeetCode #547 — Number of Provinces
# Difficulty: Medium ⚡
# Identical concept to connected components!
# Approach: Union-Find
# Time: O(n² * α(n)) | Space: O(n)

def findCircleNum(isConnected):
    n = len(isConnected)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return 0  # already same component
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return 1  # successfully merged!

    provinces = n  # start with n components
    for i in range(n):
        for j in range(i+1, n):
            if isConnected[i][j] == 1:
                provinces -= union(i, j)

    return provinces

# Test cases
print(findCircleNum(
    [[1,1,0],[1,1,0],[0,0,1]]))  # 2
print(findCircleNum(
    [[1,0,0],[0,1,0],[0,0,1]]))  # 3
print(findCircleNum(
    [[1,1,0],[1,1,1],[0,1,1]]))  # 1
