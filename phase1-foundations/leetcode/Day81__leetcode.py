# Speed — 5 min max!
def lengthOfLongestSubstring(s):
    char_idx = {}
    left = result = 0
    for right, char in enumerate(s):
        if char in char_idx and \
           char_idx[char] >= left:
            left = char_idx[char] + 1
        char_idx[char] = right
        result = max(result, right - left + 1)
    return result

print(lengthOfLongestSubstring("abcabcbb")) # 3
print(lengthOfLongestSubstring("pwwkew"))   # 3


# Phase 3 LLM Day 3 — LeetCode #76
# Minimum Window Substring
# Difficulty: Hard 🔴
# Approach: Sliding window + counter
# Time: O(n) | Space: O(k)

from collections import Counter

def minWindow(s, t):
    if not t or not s: return ""

    need = Counter(t)
    missing = len(t)  # chars still needed
    best_left = best_right = 0
    best_len = float('inf')
    left = 0

    for right, char in enumerate(s, 1):
        # Add right char to window
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        # Window valid — try to shrink!
        if missing == 0:
            # Remove unnecessary left chars
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1

            # Update best window
            if right - left < best_len:
                best_len = right - left
                best_left, best_right = left, right

            # Shrink by 1 to find next window
            need[s[left]] += 1
            missing += 1
            left += 1

    return s[best_left:best_right] \
           if best_len < float('inf') else ""

print(minWindow("ADOBECODEBANC","ABC"))  # "BANC"
print(minWindow("a","a"))               # "a"
print(minWindow("a","aa"))              # ""
