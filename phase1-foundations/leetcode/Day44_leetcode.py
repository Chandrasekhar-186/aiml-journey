# Phase 2 Day 14 — LeetCode #452 Min Arrows
# Difficulty: Medium ⚡
# Approach: Greedy interval scheduling
# Time: O(n log n) | Space: O(1)

def findMinArrowShots(points):
    # Sort by END point
    points.sort(key=lambda x: x[1])
    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        # Current balloon not burst by arrow
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end  # shoot at end

    return arrows

print(findMinArrowShots(
    [[10,16],[2,8],[1,6],[7,12]]))  # 2
print(findMinArrowShots(
    [[1,2],[3,4],[5,6],[7,8]]))     # 4
print(findMinArrowShots(
    [[1,2],[2,3],[3,4],[4,5]]))     # 2



# Phase 2 Day 14 — LeetCode #57 Insert Interval
# Difficulty: Medium ⚡
# Approach: Linear scan + merge
# Time: O(n) | Space: O(n)

def insert(intervals, newInterval):
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals ending before new starts
    while i < n and \
          intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Merge all overlapping intervals
    while i < n and \
          intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0],
                              intervals[i][0])
        newInterval[1] = max(newInterval[1],
                              intervals[i][1])
        i += 1
    result.append(newInterval)

    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result

print(insert([[1,3],[6,9]], [2,5]))
# [[1,5],[6,9]]
print(insert(
    [[1,2],[3,5],[6,7],[8,10],[12,16]],
    [4,8]))
# [[1,2],[3,10],[12,16]]
