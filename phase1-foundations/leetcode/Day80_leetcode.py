# Speed — 5 min max!
def mergeAlternately(word1, word2):
    result = []
    i = j = 0
    while i < len(word1) and j < len(word2):
        result.append(word1[i])
        result.append(word2[j])
        i += 1; j += 1
    return ''.join(result) + \
           word1[i:] + word2[j:]

print(mergeAlternately("abc","pqr"))  # "apbqcr"
print(mergeAlternately("ab","pqrs"))  # "apbqrs"

# Speed revisit — 8 min max!
from collections import Counter
def findAnagrams(s, p):
    if len(p) > len(s): return []
    p_count = Counter(p)
    window = Counter(s[:len(p)])
    result = [0] if window == p_count else []
    for i in range(len(p), len(s)):
        window[s[i]] += 1
        old = s[i-len(p)]
        window[old] -= 1
        if window[old] == 0: del window[old]
        if window == p_count:
            result.append(i-len(p)+1)
    return result

print(findAnagrams("cbaebabacd","abc"))  # [0,6]

# Phase 3 LLM Day 2 — LeetCode #567
# Permutation in String
# Difficulty: Medium ⚡
# Approach: Sliding window + counter
# Time: O(n) | Space: O(1)

from collections import Counter

def checkInclusion(s1, s2):
    if len(s1) > len(s2): return False
    s1_count = Counter(s1)
    window = Counter(s2[:len(s1)])

    if window == s1_count: return True

    for i in range(len(s1), len(s2)):
        window[s2[i]] += 1
        old = s2[i - len(s1)]
        window[old] -= 1
        if window[old] == 0: del window[old]
        if window == s1_count: return True

    return False

print(checkInclusion("ab","eidbaooo"))  # True
print(checkInclusion("ab","eidboaoo"))  # False
