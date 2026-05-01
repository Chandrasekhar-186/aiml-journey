#Problem 1: LeetCode #139 — Word Break speed challenge — solve in 5 min! (5 min)
#python# Speed challenge — 5 minutes max!
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
  
#Problem 2: LeetCode #322 — Coin Change speed challenge — solve in 5 min! (5 min)
#python# Speed challenge — 5 minutes max!
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for amt in range(1, amount+1):
        for coin in coins:
            if coin <= amt:
                dp[amt] = min(dp[amt],
                               dp[amt-coin]+1)
    return dp[amount] if dp[amount] != float('inf') \
           else -1
  
#Problem 3: LeetCode #300 — LIS speed challenge — solve in 5 min! (5 min)
#python# Speed challenge — 5 minutes max!
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)
  
#Problem 4: LeetCode #994 — Rotting Oranges speed challenge — solve in 8 min! (8 min)
#python# Speed challenge — 8 minutes max!
from collections import deque
def orangesRotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r,c,0))
            elif grid[r][c] == 1:
                fresh += 1
    if fresh == 0: return 0
    time = 0
    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        pass  # complete this!
    # YOUR FULL SOLUTION HERE
