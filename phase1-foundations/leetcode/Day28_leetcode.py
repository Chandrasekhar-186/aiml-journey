# Day 28 — LeetCode #124 Binary Tree Max Path Sum
# Difficulty: Hard 🔴
# Approach: DFS — track max at each node
# Time: O(n) | Space: O(h)

def maxPathSum(root):
    max_sum = [float('-inf')]

    def dfs(node):
        if not node:
            return 0

        # Max gain from left and right
        # Use max(0, ...) to ignore negative paths!
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))

        # Path through current node
        path_sum = node.val + left + right
        max_sum[0] = max(max_sum[0], path_sum)

        # Return max single-path gain to parent
        return node.val + max(left, right)

    dfs(root)
    return max_sum[0]


# Day 28 — LeetCode #647 Palindromic Substrings
# Difficulty: Medium ⚡
# Approach: Expand around center
# Time: O(n²) | Space: O(1)

def countSubstrings(s):
    count = 0

    def expand(left, right):
        nonlocal count
        while (left >= 0 and
               right < len(s) and
               s[left] == s[right]):
            count += 1
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)    # odd length
        expand(i, i+1)  # even length

    return count

print(countSubstrings("abc"))   # 3
print(countSubstrings("aaa"))   # 6
print(countSubstrings("racecar"))  # 10
