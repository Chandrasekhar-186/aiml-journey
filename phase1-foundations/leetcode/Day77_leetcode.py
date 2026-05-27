# Phase 3 CV Day 6 — LeetCode #48
# Rotate Image 90° clockwise
# Difficulty: Medium ⚡
# Approach: Transpose + reverse rows
# Time: O(n²) | Space: O(1)

def rotate(matrix):
    n = len(matrix)
    # Step 1: Transpose (swap i,j with j,i)
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = \
                matrix[j][i], matrix[i][j]
    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()

m = [[1,2,3],[4,5,6],[7,8,9]]
rotate(m)
print(m)  # [[7,4,1],[8,5,2],[9,6,3]]


# Phase 3 CV Day 6 — LeetCode #54
# Spiral Matrix
# Difficulty: Medium ⚡
# Approach: Shrink boundaries
# Time: O(m*n) | Space: O(1)

def spiralOrder(matrix):
    result = []
    top = left = 0
    bottom = len(matrix) - 1
    right = len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Left → Right (top row)
        for c in range(left, right+1):
            result.append(matrix[top][c])
        top += 1

        # Top → Bottom (right col)
        for r in range(top, bottom+1):
            result.append(matrix[r][right])
        right -= 1

        # Right → Left (bottom row)
        if top <= bottom:
            for c in range(right, left-1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1

        # Bottom → Top (left col)
        if left <= right:
            for r in range(bottom, top-1, -1):
                result.append(matrix[r][left])
            left += 1

    return result

print(spiralOrder(
    [[1,2,3],[4,5,6],[7,8,9]]))
# [1,2,3,6,9,8,7,4,5]
print(spiralOrder(
    [[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
# [1,2,3,4,8,12,11,10,9,5,6,7]
