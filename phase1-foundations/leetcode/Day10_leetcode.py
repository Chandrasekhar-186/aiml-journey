# Day 10 — LeetCode #200 Number of Islands
# Difficulty: Medium ⚡
# Approach: DFS Graph traversal
# Time: O(m*n) | Space: O(m*n)

def numIslands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        # Out of bounds or water → stop
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            grid[r][c] == '0'):
            return
        grid[r][c] = '0'  # mark visited!
        # Explore all 4 directions
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)   # sink the island
                count += 1  # found one island!

    return count

# Test
grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]
print(numIslands(grid))  # 3

# Day 10 — LeetCode #733 Flood Fill
# Difficulty: Easy
# Approach: DFS — same pattern as islands!
# Time: O(m*n) | Space: O(m*n)

def floodFill(image, sr, sc, color):
    original = image[sr][sc]
    if original == color:
        return image

    def dfs(r, c):
        if (r < 0 or r >= len(image) or
            c < 0 or c >= len(image[0]) or
            image[r][c] != original):
            return
        image[r][c] = color
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)

    dfs(sr, sc)
    return image
