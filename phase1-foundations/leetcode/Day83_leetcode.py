# 2 minutes max!
def reverseString(s):
    l, r = 0, len(s)-1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1; r -= 1

# 5 minutes max!
def isPalindrome(s):
    l, r = 0, len(s)-1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1; r -= 1
    return True

print(isPalindrome("A man, a plan, a canal: Panama")) # True
print(isPalindrome("race a car"))  # False


# Phase 3 LLM Day 5 — LeetCode #11
# Container with Most Water
# Difficulty: Medium ⚡
# Approach: Two pointers
# Time: O(n) | Space: O(1)

def maxArea(height):
    l, r = 0, len(height) - 1
    result = 0
    while l < r:
        area = (r - l) * min(height[l],
                               height[r])
        result = max(result, area)
        # Move the shorter pointer!
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return result

print(maxArea([1,8,6,2,5,4,8,3,7]))  # 49
print(maxArea([1,1]))                  # 1
