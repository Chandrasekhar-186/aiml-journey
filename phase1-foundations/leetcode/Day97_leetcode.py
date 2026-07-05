# Phase 5 Day 2 — LeetCode #994
# Rotting Oranges
# Difficulty: Medium ⚡
# Approach: Multi-source BFS
# Time: O(m*n) | Space: O(m*n)

from collections import deque

def orangesRotting(grid):
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0

    # Find all rotten oranges + count fresh
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 2:
                q.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0: return 0

    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    minutes = 0

    while q:
        r, c, t = q.popleft()
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if (0<=nr<m and 0<=nc<n and
                    grid[nr][nc] == 1):
                grid[nr][nc] = 2
                fresh -= 1
                minutes = max(minutes, t+1)
                q.append((nr, nc, t+1))

    return minutes if fresh == 0 else -1

print(orangesRotting(
    [[2,1,1],[1,1,0],[0,1,1]]))  # 4
print(orangesRotting(
    [[2,1,1],[0,1,1],[1,0,1]]))  # -1
print(orangesRotting([[0,2]]))   # 0


# Speed — 5 min max! Classic DFS/BFS
def floodFill(image, sr, sc, color):
    orig = image[sr][sc]
    if orig == color: return image

    def dfs(r, c):
        if (r<0 or r>=len(image) or
            c<0 or c>=len(image[0]) or
            image[r][c] != orig):
            return
        image[r][c] = color
        dfs(r+1,c); dfs(r-1,c)
        dfs(r,c+1); dfs(r,c-1)

    dfs(sr, sc)
    return image

print(floodFill(
    [[1,1,1],[1,1,0],[1,0,1]],
    1, 1, 2))
# [[2,2,2],[2,2,0],[2,0,1]]
