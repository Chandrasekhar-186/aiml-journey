# Phase 3 CV Day 7 — LeetCode #73
# Set Matrix Zeroes
# Difficulty: Medium ⚡
# Approach: Use first row/col as markers
# Time: O(m*n) | Space: O(1)

def setZeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(
        matrix[0][j] == 0 for j in range(n)
    )
    first_col_zero = any(
        matrix[i][0] == 0 for i in range(m)
    )

    # Use first row/col as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # Zero out based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or \
               matrix[0][j] == 0:
                matrix[i][j] = 0

    # Handle first row/col
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0

m = [[1,1,1],[1,0,1],[1,1,1]]
setZeroes(m)
print(m)  # [[1,0,1],[0,0,0],[1,0,1]]


# Phase 3 CV Day 7 — LeetCode #36
# Valid Sudoku
# Difficulty: Medium ⚡
# Approach: HashSet for rows/cols/boxes
# Time: O(81) = O(1) | Space: O(81) = O(1)

def isValidSudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue

            box_idx = (r//3)*3 + c//3

            if (val in rows[r] or
                val in cols[c] or
                val in boxes[box_idx]):
                return False

            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)

    return True
