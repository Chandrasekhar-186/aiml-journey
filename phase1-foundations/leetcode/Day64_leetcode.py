# Speed — 5 min max!
def floodFill(image, sr, sc, color):
    orig = image[sr][sc]
    if orig == color: return image

    def dfs(r, c):
        if (r < 0 or r >= len(image) or
                c < 0 or c >= len(image[0]) or
                image[r][c] != orig):
            return
        image[r][c] = color
        dfs(r+1,c); dfs(r-1,c)
        dfs(r,c+1); dfs(r,c-1)

    dfs(sr, sc)
    return image

print(floodFill(
    [[1,1,1],[1,1,0],[1,0,1]],
    1, 1, 2))  # [[2,2,2],[2,2,0],[2,0,1]]


# Phase 3 Day 6 — #417 Speed revisit
# Target: solve in 12 minutes from memory!
from collections import deque

def pacificAtlantic(heights):
    rows, cols = len(heights), len(heights[0])
    pac, atl = set(), set()

    def bfs(starts, visited):
        queue = deque(starts)
        visited.update(starts)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(1,0),(-1,0),
                            (0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < rows and
                    0 <= nc < cols and
                    (nr,nc) not in visited and
                    heights[nr][nc] >=
                    heights[r][c]):
                    visited.add((nr,nc))
                    queue.append((nr,nc))

    bfs([(0,c) for c in range(cols)] +
        [(r,0) for r in range(rows)], pac)
    bfs([(rows-1,c) for c in range(cols)] +
        [(r,cols-1) for r in range(rows)], atl)

    return [[r,c] for r in range(rows)
            for c in range(cols)
            if (r,c) in pac and (r,c) in atl]

print(len(pacificAtlantic(
    [[1,2,2,3,5],[3,2,3,4,4],
     [2,4,5,3,1],[6,7,1,4,5],
     [5,1,1,2,4]])))  # 7
