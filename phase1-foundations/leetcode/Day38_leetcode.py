# Phase 2 Day 8 — LeetCode #399 Evaluate Division
# Difficulty: Medium ⚡
# Approach: Graph BFS / Union-Find with weights
# Time: O((V+E) * Q) | Space: O(V+E)

from collections import defaultdict, deque

def calcEquation(equations, values, queries):
    # Build weighted graph
    graph = defaultdict(dict)
    for (a, b), val in zip(equations, values):
        graph[a][b] = val
        graph[b][a] = 1.0 / val

    def bfs(src, dst):
        if src not in graph or dst not in graph:
            return -1.0
        if src == dst:
            return 1.0

        visited = set()
        queue = deque([(src, 1.0)])
        visited.add(src)

        while queue:
            node, product = queue.popleft()
            if node == dst:
                return product
            for neighbor, weight in \
                    graph[node].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(
                        (neighbor,
                         product * weight)
                    )
        return -1.0

    return [bfs(s, t) for s, t in queries]

equations = [["a","b"],["b","c"]]
values = [2.0, 3.0]
queries = [["a","c"],["b","a"],
           ["a","e"],["a","a"],["x","x"]]
print(calcEquation(equations, values, queries))
# [6.0, 0.5, -1.0, 1.0, -1.0]


# Phase 2 Day 8 — LeetCode #1584 Min Cost MST
# Difficulty: Medium ⚡
# Approach: Prim's MST algorithm
# Time: O(n² log n) | Space: O(n)

import heapq

def minCostConnectPoints(points):
    n = len(points)
    visited = set()
    heap = [(0, 0)]  # (cost, point_index)
    total = 0

    while len(visited) < n:
        cost, i = heapq.heappop(heap)

        if i in visited:
            continue

        visited.add(i)
        total += cost

        # Add edges to all unvisited points
        for j in range(n):
            if j not in visited:
                dist = (abs(points[i][0] -
                            points[j][0]) +
                        abs(points[i][1] -
                            points[j][1]))
                heapq.heappush(heap, (dist, j))

    return total

print(minCostConnectPoints(
    [[0,0],[2,2],[3,10],[5,2],[7,0]]))  # 20
print(minCostConnectPoints(
    [[3,12],[-2,5],[-4,1]]))           # 18
