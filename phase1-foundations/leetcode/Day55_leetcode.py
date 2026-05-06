# Phase 2 Day 25 — LeetCode #394 Decode String
# Difficulty: Medium ⚡
# Approach: Stack
# Time: O(max_k * n) | Space: O(n)

def decodeString(s):
    stack = []  # stores (repeat_count, built_string)
    current_str = ""
    current_num = 0

    for char in s:
        if char.isdigit():
            current_num = current_num*10 + int(char)
        elif char == '[':
            # Push current state
            stack.append((current_num, current_str))
            current_str = ""
            current_num = 0
        elif char == ']':
            # Pop and repeat
            num, prev_str = stack.pop()
            current_str = prev_str + num * current_str
        else:
            current_str += char

    return current_str

print(decodeString("3[a]2[bc]"))    # "aaabcbc"
print(decodeString("3[a2[c]]"))     # "accaccacc"
print(decodeString("2[abc]3[cd]ef"))# "abcabccdcdcdef"


# Speed challenge — 8 min max!
def dailyTemperatures(temps):
    stack = []
    result = [0] * len(temps)
    for i, t in enumerate(temps):
        while stack and t > temps[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    return result

print(dailyTemperatures(
    [73,74,75,71,69,72,76,73]))
# [1,1,4,2,1,1,0,0]
