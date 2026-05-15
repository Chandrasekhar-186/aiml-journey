# 3 minutes max — should be instant now!
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit,
                          price - min_price)
    return max_profit

print(maxProfit([7,1,5,3,6,4]))  # 5
print(maxProfit([7,6,4,3,1]))    # 0

# 3 minutes max!
def maxSubArray(nums):
    cur = res = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        res = max(res, cur)
    return res

print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4])) # 6
print(maxSubArray([1]))                       # 1
print(maxSubArray([5,4,-1,7,8]))             # 23
