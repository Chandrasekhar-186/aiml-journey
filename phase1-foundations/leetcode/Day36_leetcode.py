# Phase 2 Day 6 — LeetCode #621 Task Scheduler
# Difficulty: Medium ⚡
# Approach: Greedy with frequency counting
# Time: O(n) | Space: O(1)

from collections import Counter

def leastInterval(tasks, n):
    count = Counter(tasks)
    max_freq = max(count.values())

    # How many tasks have max frequency?
    max_count = sum(
        1 for v in count.values()
        if v == max_freq
    )

    # Formula: max of actual tasks vs
    # minimum slots needed
    # Slots = (max_freq - 1) * (n + 1) + max_count
    slots = (max_freq - 1) * (n + 1) + max_count

    return max(len(tasks), slots)

print(leastInterval(
    ["A","A","A","B","B","B"], 2))  # 8
print(leastInterval(
    ["A","A","A","B","B","B"], 0))  # 6
print(leastInterval(
    ["A","A","A","A","A","A","B","C","D",
     "E","F","G"], 2))              # 16


# Phase 2 Day 6 — LeetCode #332 Itinerary
# Difficulty: Hard 🔴
# Approach: Hierholzer's Eulerian path
# Time: O(E log E) | Space: O(E)

from collections import defaultdict
import heapq

def findItinerary(tickets):
    graph = defaultdict(list)

    # Build min-heap adjacency list
    for src, dst in tickets:
        heapq.heappush(graph[src], dst)

    result = []

    def dfs(airport):
        while graph[airport]:
            next_dest = heapq.heappop(
                graph[airport]
            )
            dfs(next_dest)
        result.append(airport)

    dfs("JFK")
    return result[::-1]

print(findItinerary(
    [["MUC","LHR"],["JFK","MUC"],
     ["SFO","SJC"],["LHR","SFO"]]))
# ["JFK","MUC","LHR","SFO","SJC"]

print(findItinerary(
    [["JFK","SFO"],["JFK","ATL"],
     ["SFO","ATL"],["ATL","JFK"],
     ["ATL","SFO"]]))
# ["JFK","ATL","JFK","SFO","ATL","SFO"]
