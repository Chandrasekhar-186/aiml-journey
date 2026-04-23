# Phase 2 Day 7 — LeetCode #309 Stock Cooldown
# Difficulty: Medium ⚡
# Approach: State machine DP
# Time: O(n) | Space: O(1)

def maxProfit(prices):
    # 3 states:
    # held:     holding stock
    # sold:     just sold (next day = cooldown)
    # rest:     cooldown or waiting

    held = float('-inf')  # max profit holding
    sold = 0              # max profit after sell
    rest = 0              # max profit resting

    for price in prices:
        prev_held = held
        prev_sold = sold
        prev_rest = rest

        # Buy: must be in rest state
        held = max(prev_held,
                   prev_rest - price)
        # Sell: must be holding
        sold = prev_held + price
        # Rest: was sold (cooldown) or resting
        rest = max(prev_rest, prev_sold)

    return max(sold, rest)

print(maxProfit([1,2,3,0,2]))   # 3
print(maxProfit([1]))            # 0
print(maxProfit([1,2]))          # 1




# Phase 2 Day 7 — LeetCode #1091 Binary Matrix
# Difficulty: Medium ⚡
# Approach: BFS shortest path
# Time: O(n²) | Space: O(n²)

from collections import deque

def shortestPathBinaryMatrix(grid):
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    queue = deque([(0, 0, 1)])  # row, col, dist
    grid[0][0] = 1  # mark visited

    directions = [(-1,-1),(-1,0),(-1,1),
                  (0,-1),        (0,1),
                  (1,-1), (1,0), (1,1)]

    while queue:
        r, c, dist = queue.popleft()

        if r == n-1 and c == n-1:
            return dist

        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if (0 <= nr < n and
                0 <= nc < n and
                grid[nr][nc] == 0):
                grid[nr][nc] = 1  # mark visited
                queue.append((nr, nc, dist+1))

    return -1

print(shortestPathBinaryMatrix(
    [[0,1],[1,0]]))                    # 2
print(shortestPathBinaryMatrix(
    [[0,0,0],[1,1,0],[1,1,0]]))        # 4
