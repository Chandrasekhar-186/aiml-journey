# Day 17 — LeetCode #76 Minimum Window Substring
# Difficulty: Hard 🔴
# Approach: Sliding Window + frequency maps
# Time: O(n) | Space: O(k)

from collections import Counter

def minWindow(s, t):
    if not t or not s:
        return ""

    need = Counter(t)      # chars needed
    have = {}              # chars in window
    formed = 0             # unique chars satisfied
    required = len(need)   # unique chars needed

    left = 0
    min_len = float('inf')
    result = ""

    for right in range(len(s)):
        char = s[right]
        have[char] = have.get(char, 0) + 1

        # Check if this char's requirement is met
        if (char in need and
                have[char] == need[char]):
            formed += 1

        # Try to shrink window from left
        while formed == required:
            # Update result
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right+1]

            # Remove leftmost char
            left_char = s[left]
            have[left_char] -= 1
            if (left_char in need and
                    have[left_char] < need[left_char]):
                formed -= 1
            left += 1

    return result

# Test cases
print(minWindow("ADOBECODEBANC", "ABC"))  # "BANC"
print(minWindow("a", "a"))                # "a"
print(minWindow("a", "aa"))               # ""

# Day 17 — LeetCode #46 Permutations
# Difficulty: Medium ⚡
# Approach: Backtracking
# Time: O(n! * n) | Space: O(n)

def permute(nums):
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current,
                       remaining[:i] +
                       remaining[i+1:])
            current.pop()  # undo choice!

    backtrack([], nums)
    return result

print(permute([1,2,3]))  # 6 permutations
print(len(permute([1,2,3,4])))  # 24
