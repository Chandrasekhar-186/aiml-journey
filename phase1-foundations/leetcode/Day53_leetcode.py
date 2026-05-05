# Phase 2 Day 23 — LeetCode #329
# Longest Increasing Path in Matrix
# Difficulty: Hard 🔴
# Approach: DFS + memoization (top-down DP)
# Time: O(m*n) | Space: O(m*n)

def longestIncreasingPath(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]

        best = 1
        for dr, dc in [(0,1),(0,-1),
                        (1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if (0 <= nr < m and
                0 <= nc < n and
                matrix[nr][nc] > matrix[r][c]):
                best = max(best,
                            1 + dfs(nr, nc))

        memo[(r, c)] = best
        return best

    return max(dfs(r, c)
               for r in range(m)
               for c in range(n))

print(longestIncreasingPath(
    [[9,9,4],
     [6,6,8],
     [2,1,1]]))   # 4
print(longestIncreasingPath(
    [[3,4,5],
     [3,2,6],
     [2,2,1]]))   # 4

# Speed challenge — 10 min max!
# Backtracking pattern

def generateParenthesis(n):
    result = []

    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s+'(', open_count+1,
                       close_count)
        if close_count < open_count:
            backtrack(s+')', open_count,
                       close_count+1)

    backtrack("", 0, 0)
    return result

print(generateParenthesis(3))  # 5 combos
print(generateParenthesis(1))  # ["()"]
