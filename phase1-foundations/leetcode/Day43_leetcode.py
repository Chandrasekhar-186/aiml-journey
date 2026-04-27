# Phase 2 Day 13 — LeetCode #213 House Robber II
# Difficulty: Medium ⚡
# Approach: Run House Robber I twice
# Time: O(n) | Space: O(1)

def rob(nums):
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses):
        prev2 = prev1 = 0
        for h in houses:
            prev2, prev1 = prev1, max(prev1,
                                       prev2 + h)
        return prev1

    # Can't rob both first AND last house
    # → Run twice: skip last OR skip first
    return max(
        rob_linear(nums[:-1]),  # skip last
        rob_linear(nums[1:])    # skip first
    )

print(rob([2,3,2]))      # 3
print(rob([1,2,3,1]))    # 4
print(rob([1,2,3]))      # 3


# Phase 2 Day 13 — LeetCode #337 House Robber III
# Difficulty: Medium ⚡
# Approach: Tree DP — rob or skip each node
# Time: O(n) | Space: O(h)

def rob(root):
    def dfs(node):
        if not node:
            return (0, 0)  # (rob, skip)

        left_rob, left_skip = dfs(node.left)
        right_rob, right_skip = dfs(node.right)

        # Rob this node: can't rob children
        rob_curr = (node.val +
                    left_skip + right_skip)
        # Skip this node: take best of children
        skip_curr = (max(left_rob, left_skip) +
                     max(right_rob, right_skip))

        return (rob_curr, skip_curr)

    rob_root, skip_root = dfs(root)
    return max(rob_root, skip_root)
