# Phase 2 Day 20 — LeetCode #678
# Valid Parenthesis String
# Difficulty: Medium ⚡
# Approach: Greedy (track min/max open)
# Time: O(n) | Space: O(1)

def checkValidString(s):
    # Track range of possible open counts
    min_open = max_open = 0

    for char in s:
        if char == '(':
            min_open += 1
            max_open += 1
        elif char == ')':
            min_open -= 1
            max_open -= 1
        else:  # '*' can be (, ), or empty
            min_open -= 1  # treat as )
            max_open += 1  # treat as (

        if max_open < 0:
            return False  # too many )
        min_open = max(min_open, 0)

    return min_open == 0

print(checkValidString("()"))      # True
print(checkValidString("(*)"))     # True
print(checkValidString("(*))"))    # True
print(checkValidString("(((*)"))   # False


# Phase 2 Day 20 — LeetCode #986
# Interval List Intersections
# Difficulty: Medium ⚡
# Approach: Two pointers
# Time: O(m+n) | Space: O(m+n)

def intervalIntersection(firstList, secondList):
    result = []
    i = j = 0

    while i < len(firstList) and \
          j < len(secondList):
        # Find intersection
        lo = max(firstList[i][0],
                  secondList[j][0])
        hi = min(firstList[i][1],
                  secondList[j][1])

        if lo <= hi:
            result.append([lo, hi])

        # Move pointer with smaller end
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1

    return result

print(intervalIntersection(
    [[0,2],[5,10],[13,23],[24,25]],
    [[1,5],[8,12],[15,24],[25,26]]))
# [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
