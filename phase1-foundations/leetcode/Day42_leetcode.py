# Phase 2 Day 12 — LeetCode #130 Surrounded Regions
# Difficulty: Medium ⚡
# Approach: DFS from borders
# Time: O(m*n) | Space: O(m*n)

def solve(board):
    if not board: return
    m, n = len(board), len(board[0])

    def dfs(r, c):
        if (r < 0 or r >= m or
            c < 0 or c >= n or
            board[r][c] != 'O'):
            return
        board[r][c] = 'S'  # safe!
        dfs(r+1,c); dfs(r-1,c)
        dfs(r,c+1); dfs(r,c-1)

    # Mark border-connected 'O's as safe
    for r in range(m):
        dfs(r, 0); dfs(r, n-1)
    for c in range(n):
        dfs(0, c); dfs(m-1, c)

    # Flip remaining 'O' → 'X', restore 'S' → 'O'
    for r in range(m):
        for c in range(n):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'S':
                board[r][c] = 'O'

board = [["X","X","X","X"],
         ["X","O","O","X"],
         ["X","X","O","X"],
         ["X","O","X","X"]]
solve(board)
print(board)
# [["X","X","X","X"],
#  ["X","X","X","X"],
#  ["X","X","X","X"],
#  ["X","O","X","X"]]


# Phase 2 Day 12 — LeetCode #417 revisit
# SPEED CHALLENGE: solve in under 10 minutes!
# Difficulty: Medium ⚡
# Approach: Reverse BFS from both oceans

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

heights = [[1,2,2,3,5],
           [3,2,3,4,4],
           [2,4,5,3,1],
           [6,7,1,4,5],
           [5,1,1,2,4]]
print(len(pacificAtlantic(heights)))  # 7
