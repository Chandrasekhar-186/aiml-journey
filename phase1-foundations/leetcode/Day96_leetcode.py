# Phase 5 Day 1 — LeetCode #207
# Course Schedule
# Difficulty: Medium ⚡
# Approach: DFS cycle detection
# Time: O(V+E) | Space: O(V+E)

def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[b].append(a)

    # 0=unvisited, 1=visiting, 2=visited
    state = [0] * numCourses

    def has_cycle(node):
        if state[node] == 1: return True  # cycle!
        if state[node] == 2: return False

        state[node] = 1  # mark visiting
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        state[node] = 2  # mark done
        return False

    return not any(
        has_cycle(i)
        for i in range(numCourses)
        if state[i] == 0
    )

print(canFinish(2, [[1,0]]))        # True
print(canFinish(2, [[1,0],[0,1]]))  # False

# Phase 5 Day 1 — LeetCode #210
# Course Schedule II — topological sort!
# Difficulty: Medium ⚡

def findOrder(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[b].append(a)

    state = [0] * numCourses
    order = []

    def dfs(node):
        if state[node] == 1: return False
        if state[node] == 2: return True
        state[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        state[node] = 2
        order.append(node)  # post-order!
        return True

    for i in range(numCourses):
        if state[i] == 0:
            if not dfs(i):
                return []

    return order[::-1]  # reverse post-order!

print(findOrder(4,
    [[1,0],[2,0],[3,1],[3,2]]))  # [0,1,2,3]
print(findOrder(2, [[0,1]]))     # [1,0]
