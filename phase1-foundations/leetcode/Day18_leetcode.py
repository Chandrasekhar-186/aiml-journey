# Day 18 — LeetCode #207 Course Schedule
# Difficulty: Medium ⚡
# Approach: DFS cycle detection (topological sort)
# Time: O(V+E) | Space: O(V+E)

def canFinish(numCourses, prerequisites):
    # Build adjacency list
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 0=unvisited, 1=visiting, 2=visited
    state = [0] * numCourses

    def has_cycle(node):
        if state[node] == 1:  # cycle detected!
            return True
        if state[node] == 2:  # already processed
            return False

        state[node] = 1  # mark as visiting
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        state[node] = 2  # mark as done
        return False

    for i in range(numCourses):
        if has_cycle(i):
            return False
    return True

# Test cases
print(canFinish(2, [[1,0]]))           # True
print(canFinish(2, [[1,0],[0,1]]))     # False (cycle!)
print(canFinish(4, [[1,0],[2,1],[3,2]]))  # True


# Day 18 — LeetCode #79 Word Search
# Difficulty: Medium ⚡
# Approach: Backtracking DFS
# Time: O(m*n*4^L) | Space: O(L)

def exist(board, word):
    rows, cols = len(board), len(board[0])

    def dfs(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            board[r][c] != word[idx]):
            return False

        # Mark visited
        temp = board[r][c]
        board[r][c] = '#'

        # Explore all 4 directions
        found = (dfs(r+1,c,idx+1) or
                 dfs(r-1,c,idx+1) or
                 dfs(r,c+1,idx+1) or
                 dfs(r,c-1,idx+1))

        # Restore (backtrack!)
        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False

board = [["A","B","C","E"],
         ["S","F","C","S"],
         ["A","D","E","E"]]
print(exist(board, "ABCCED"))  # True
print(exist(board, "SEE"))     # True
print(exist(board, "ABCB"))    # False
