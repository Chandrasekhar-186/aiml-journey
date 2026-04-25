# Phase 2 Day 9 — LeetCode #778 Swim Rising Water
# Difficulty: Hard 🔴
# Approach: Dijkstra (min-heap)
# Time: O(n² log n) | Space: O(n²)

import heapq

def swimInWater(grid):
    n = len(grid)
    heap = [(grid[0][0], 0, 0)]
    visited = set()
    visited.add((0, 0))

    directions = [(0,1),(0,-1),(1,0),(-1,0)]

    while heap:
        t, r, c = heapq.heappop(heap)

        if r == n-1 and c == n-1:
            return t  # minimum time!

        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if (0 <= nr < n and
                0 <= nc < n and
                (nr, nc) not in visited):
                visited.add((nr, nc))
                # Time = max along path
                heapq.heappush(heap, (
                    max(t, grid[nr][nc]),
                    nr, nc
                ))

    return -1

print(swimInWater([[0,2],[1,3]]))           # 3
print(swimInWater(
    [[0,1,2,3,4],
     [24,23,22,21,5],
     [12,13,14,15,16],
     [11,17,18,19,20],  # corrected
     [10,9,8,7,6]]))    # 16

# Phase 2 Day 9 — LeetCode #997 Town Judge
# Difficulty: Easy ✅
# Approach: In-degree vs out-degree
# Time: O(V+E) | Space: O(V)

def findJudge(n, trust):
    in_degree = [0] * (n + 1)
    out_degree = [0] * (n + 1)

    for a, b in trust:
        out_degree[a] += 1
        in_degree[b] += 1

    for i in range(1, n + 1):
        # Judge trusts nobody + trusted by all
        if (out_degree[i] == 0 and
                in_degree[i] == n - 1):
            return i
    return -1

print(findJudge(2, [[1,2]]))             # 2
print(findJudge(3, [[1,3],[2,3]]))       # 3
print(findJudge(3, [[1,3],[2,3],[3,1]])) # -1
