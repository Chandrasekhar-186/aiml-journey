# Phase 2 Day 19 — LeetCode #376 Wiggle Subseq
# Difficulty: Medium ⚡
# Approach: Greedy
# Time: O(n) | Space: O(1)

def wiggleMaxLength(nums):
    if len(nums) < 2:
        return len(nums)

    up = down = 1

    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            up = down + 1
        elif nums[i] < nums[i-1]:
            down = up + 1

    return max(up, down)

print(wiggleMaxLength([1,7,4,9,2,5]))   # 6
print(wiggleMaxLength([1,17,5,10,13,15,10,5,16,8]))  # 7
print(wiggleMaxLength([1,2,3,4,5,6,7]))  # 2


# Phase 2 Day 19 — LeetCode #406
# Queue Reconstruction by Height
# Difficulty: Medium ⚡
# Approach: Sort + greedy insertion
# Time: O(n²) | Space: O(n)

def reconstructQueue(people):
    # Sort: tallest first, fewer-in-front first
    people.sort(key=lambda x: (-x[0], x[1]))
    result = []
    for person in people:
        # Insert at position k
        result.insert(person[1], person)
    return result

print(reconstructQueue(
    [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]))
# [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
