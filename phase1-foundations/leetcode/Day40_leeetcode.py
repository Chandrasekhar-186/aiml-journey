# Phase 2 Day 10 — LeetCode #133 Clone Graph
# Difficulty: Medium ⚡
# Approach: BFS + HashMap
# Time: O(V+E) | Space: O(V)

from collections import deque

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors or []

def cloneGraph(node):
    if not node:
        return None

    cloned = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        curr = queue.popleft()
        for neighbor in curr.neighbors:
            if neighbor not in cloned:
                cloned[neighbor] = Node(
                    neighbor.val
                )
                queue.append(neighbor)
            cloned[curr].neighbors.append(
                cloned[neighbor]
            )

    return cloned[node]


# Phase 2 Day 10 — LeetCode #1514
# Path with Maximum Probability
# Difficulty: Medium ⚡
# Approach: Modified Dijkstra (max probability)
# Time: O(E log V) | Space: O(V+E)

import heapq
from collections import defaultdict

def maxProbability(n, edges, succProb,
                   start, end):
    graph = defaultdict(list)
    for (a, b), p in zip(edges, succProb):
        graph[a].append((b, p))
        graph[b].append((a, p))

    # Max-heap: negate probability!
    heap = [(-1.0, start)]
    probs = [0.0] * n
    probs[start] = 1.0

    while heap:
        neg_prob, node = heapq.heappop(heap)
        prob = -neg_prob

        if node == end:
            return prob

        if prob < probs[node]:
            continue

        for neighbor, edge_prob in graph[node]:
            new_prob = prob * edge_prob
            if new_prob > probs[neighbor]:
                probs[neighbor] = new_prob
                heapq.heappush(heap,
                    (-new_prob, neighbor))

    return 0.0

print(maxProbability(
    3, [[0,1],[1,2],[0,2]],
    [0.5,0.5,0.2], 0, 2))  # 0.25
print(maxProbability(
    3, [[0,1],[1,2],[0,2]],
    [0.5,0.5,0.3], 0, 2))  # 0.3
