# Day 02 — LeetCode #20 Valid Parentheses
# Difficulty: Easy
# Date: March 14, 2026
# Approach: Stack
# Time: O(n) | Space: O(n)

def isValid(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    
    return not stack
