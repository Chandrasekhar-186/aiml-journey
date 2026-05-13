# Speed — 8 min max! Classic DFS flood fill
def numIslands(grid):
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or
                c < 0 or c >= cols or
                grid[r][c] != '1'):
            return
        grid[r][c] = '0'  # mark visited
        dfs(r+1,c); dfs(r-1,c)
        dfs(r,c+1); dfs(r,c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count

print(numIslands(
    [["1","1","0","0","0"],
     ["1","1","0","0","0"],
     ["0","0","1","0","0"],
     ["0","0","0","1","1"]]))  # 3


# Phase 3 Day 5 — LeetCode #695
# Max Area of Island
# Difficulty: Medium ⚡
# Approach: DFS returning area count
# Time: O(m*n) | Space: O(m*n)

def maxAreaOfIsland(grid):
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if (r < 0 or r >= rows or
                c < 0 or c >= cols or
                grid[r][c] != 1):
            return 0
        grid[r][c] = 0  # mark visited
        return (1 +
                dfs(r+1,c) + dfs(r-1,c) +
                dfs(r,c+1) + dfs(r,c-1))

    return max(
        dfs(r, c)
        for r in range(rows)
        for c in range(cols)
    )

print(maxAreaOfIsland(
    [[0,0,1,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,1,1,1,0,0,0],
     [0,1,1,0,1,0,0,0,0,0,0,0,0],
     [0,1,0,0,1,1,0,0,1,0,1,0,0],
     [0,1,0,0,1,1,0,0,1,1,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,1,1,1,0,0,0],
     [0,0,0,0,0,0,0,1,1,0,0,0,0]]))  # 6
