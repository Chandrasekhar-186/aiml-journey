# Day 16 — LeetCode #417 Pacific Atlantic Water Flow
# Difficulty: Medium ⚡
# Approach: Reverse BFS from both oceans
# Time: O(m*n) | Space: O(m*n)

from collections import deque

def pacificAtlantic(heights):
    rows, cols = len(heights), len(heights[0])
    pac = set()
    atl = set()

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

    # Pacific touches top + left edges
    pac_starts = (
        [(0,c) for c in range(cols)] +
        [(r,0) for r in range(rows)]
    )
    # Atlantic touches bottom + right edges
    atl_starts = (
        [(rows-1,c) for c in range(cols)] +
        [(r,cols-1) for r in range(rows)]
    )

    bfs(pac_starts, pac)
    bfs(atl_starts, atl)

    # Cells reachable by BOTH oceans
    return [[r,c] for r in range(rows)
            for c in range(cols)
            if (r,c) in pac and (r,c) in atl]

heights = [[1,2,2,3,5],
           [3,2,3,4,4],
           [2,4,5,3,1],
           [6,7,1,4,5],
           [5,1,1,2,4]]
print(len(pacificAtlantic(heights)))  # 7 cells


# Day 16 — LeetCode #424 Longest Repeating Char
# Difficulty: Medium ⚡
# Approach: Sliding Window + frequency count
# Time: O(n) | Space: O(1)

def characterReplacement(s, k):
    count = {}
    left = 0
    max_count = 0
    result = 0

    for right in range(len(s)):
        count[s[right]] = count.get(
            s[right], 0) + 1
        max_count = max(max_count,
                         count[s[right]])

        # Window size - max_freq > k → shrink
        if (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1

        result = max(result, right - left + 1)

    return result

print(characterReplacement("ABAB", 2))   # 4
print(characterReplacement("AABABBA", 1)) # 4
