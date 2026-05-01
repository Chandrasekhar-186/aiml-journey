# Speed challenge — 10 min max!
from collections import deque

def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses

    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1

    queue = deque(
        [i for i in range(numCourses)
         if indegree[i] == 0]
    )
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1
        for next_c in graph[course]:
            indegree[next_c] -= 1
            if indegree[next_c] == 0:
                queue.append(next_c)

    return completed == numCourses

print(canFinish(2, [[1,0]]))        # True
print(canFinish(2, [[1,0],[0,1]]))  # False


# Phase 2 Day 18 — LeetCode #210
# Course Schedule II — return order!
# Difficulty: Medium ⚡
# Approach: Topological sort (BFS/Kahn's)
# Time: O(V+E) | Space: O(V+E)

from collections import deque

def findOrder(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses

    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1

    queue = deque(
        [i for i in range(numCourses)
         if indegree[i] == 0]
    )
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)
        for next_c in graph[course]:
            indegree[next_c] -= 1
            if indegree[next_c] == 0:
                queue.append(next_c)

    return order if len(order) == numCourses \
           else []

print(findOrder(4,
    [[1,0],[2,0],[3,1],[3,2]]))
# [0,1,2,3] or [0,2,1,3]
print(findOrder(2, [[1,0]]))  # [0,1]
print(findOrder(2,
    [[1,0],[0,1]]))            # [] (cycle!)
