# Speed — 5 min max! Classic DP
def rob(nums):
    prev2 = prev1 = 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1,
                                    prev2+n)
    return prev1

print(rob([1,2,3,1]))      # 4
print(rob([2,7,9,3,1]))    # 12


# Phase 4 Day 7 — LeetCode #213
# House Robber II (circular!)
# Difficulty: Medium ⚡
# Approach: Run Rob I twice
# Time: O(n) | Space: O(1)

def rob2(nums):
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses):
        prev2 = prev1 = 0
        for n in houses:
            prev2, prev1 = prev1, max(prev1,
                                       prev2+n)
        return prev1

    # Can't rob both first and last!
    # Case 1: skip last house
    # Case 2: skip first house
    # Take maximum of both cases
    return max(
        rob_linear(nums[:-1]),
        rob_linear(nums[1:])
    )

print(rob2([2,3,2]))        # 3
print(rob2([1,2,3,1]))      # 4
print(rob2([1,2,3]))        # 3
print(rob2([200,3,140,20,10]))  # 340
