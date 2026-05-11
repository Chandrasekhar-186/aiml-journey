# Speed test — this is a real Hard!
# Already solved Day 57 — now reproduce from memory

def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    half = (m + n) // 2
    lo, hi = 0, m

    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        l1 = nums1[i-1] if i > 0 \
             else float('-inf')
        r1 = nums1[i] if i < m \
             else float('inf')
        l2 = nums2[j-1] if j > 0 \
             else float('-inf')
        r2 = nums2[j] if j < n \
             else float('inf')

        if l1 <= r2 and l2 <= r1:
            if (m+n) % 2 == 1:
                return float(min(r1, r2))
            return (max(l1,l2) + min(r1,r2)) / 2.0
        elif l1 > r2:
            hi = i - 1
        else:
            lo = i + 1

print(findMedianSortedArrays([1,3],[2]))    # 2.0
print(findMedianSortedArrays([1,2],[3,4]))  # 2.5


#LeetCode #268 — Missing Number
# Speed — 3 min max!
def missingNumber(nums):
    n = len(nums)
    return n*(n+1)//2 - sum(nums)

print(missingNumber([3,0,1]))    # 2
print(missingNumber([9,6,4,2,3,5,7,0,1]))  # 8
