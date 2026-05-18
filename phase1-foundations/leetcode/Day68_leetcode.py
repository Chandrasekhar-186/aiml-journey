# Speed — 5 min max!
def hasPathSum(root, targetSum):
    if not root: return False
    if not root.left and not root.right:
        return root.val == targetSum
    return (hasPathSum(root.left,
                        targetSum - root.val) or
            hasPathSum(root.right,
                        targetSum - root.val))

# Phase 3 Day 10 — LeetCode #437
# Path Sum III
# Difficulty: Medium ⚡
# Approach: Prefix sum + HashMap (like subarray sum!)
# Time: O(n) | Space: O(n)

from collections import defaultdict

def pathSum(root, targetSum):
    # Prefix sum approach — same as subarray sum!
    prefix_count = defaultdict(int)
    prefix_count[0] = 1

    def dfs(node, curr_sum):
        if not node: return 0
        curr_sum += node.val
        # How many paths end here with targetSum?
        count = prefix_count[
            curr_sum - targetSum
        ]
        # Add current prefix to map
        prefix_count[curr_sum] += 1
        # Recurse
        count += dfs(node.left, curr_sum)
        count += dfs(node.right, curr_sum)
        # Backtrack! (remove current prefix)
        prefix_count[curr_sum] -= 1
        return count

    return dfs(root, 0)
