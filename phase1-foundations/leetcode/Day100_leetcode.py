# Speed — 5 min! Transpose + reverse rows
def rotate(matrix):
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = \
                matrix[j][i], matrix[i][j]
    # Reverse each row
    for row in matrix:
        row.reverse()

m = [[1,2,3],[4,5,6],[7,8,9]]
rotate(m)
print(m)  # [[7,4,1],[8,5,2],[9,6,3]]

# From memory — 5 min!
def productExceptSelf(nums):
    n = len(nums)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n-1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res
