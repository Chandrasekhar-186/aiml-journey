 #LeetCode #198 — House Robber speed challenge
# Speed challenge — should take < 5 min now!
def rob(nums):
    prev2 = prev1 = 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1, prev2+n)
    return prev1

# Phase 2 Day 16 — LeetCode #1048
# Longest String Chain
# Difficulty: Medium ⚡
# Approach: DP with sorting
# Time: O(n * L²) | Space: O(n)

def longestStrChain(words):
    # Sort by length — shorter words first
    words.sort(key=len)
    dp = {}  # word → longest chain ending here

    best = 1
    for word in words:
        dp[word] = 1  # chain of just itself

        # Try removing each character
        for i in range(len(word)):
            predecessor = word[:i] + word[i+1:]
            if predecessor in dp:
                dp[word] = max(dp[word],
                                dp[predecessor] + 1)
        best = max(best, dp[word])

    return best

print(longestStrChain(
    ["a","b","ba","bca","bda","bdca"]))  # 4
print(longestStrChain(
    ["xbc","pcxbcf","xb","cxbc","pcxbc"]))  # 5
print(longestStrChain(["abcd","dbqca"]))  # 1
