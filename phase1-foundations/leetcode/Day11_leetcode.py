# Day 11 — LeetCode #994 Rotting Oranges
# Difficulty: Medium ⚡
# Approach: Multi-source BFS
# Time: O(m*n) | Space: O(m*n)

from collections import deque

def orangesRotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Find all rotten oranges + count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row,col,time)
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0  # no fresh oranges!

    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    time = 0

    while queue:
        r, c, t = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and
                0 <= nc < cols and
                grid[nr][nc] == 1):
                grid[nr][nc] = 2   # rot it!
                fresh -= 1
                time = max(time, t + 1)
                queue.append((nr, nc, t + 1))

    return time if fresh == 0 else -1

# Test
print(orangesRotting([[2,1,1],[1,1,0],[0,1,1]])) # 4
print(orangesRotting([[2,1,1],[0,1,1],[1,0,1]])) # -1


# Day 11 — LeetCode #543 Diameter of Binary Tree
# Difficulty: Easy
# Approach: DFS — track max path through each node
# Time: O(n) | Space: O(h)

def diameterOfBinaryTree(root):
    max_diameter = [0]  # use list for closure

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        # Update diameter through this node
        max_diameter[0] = max(max_diameter[0],
                               left + right)
        return 1 + max(left, right)

    dfs(root)
    return max_diameter[0]
