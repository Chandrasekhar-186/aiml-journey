class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        for price in prices:
            # update minimum price seen so far
            min_price = min(min_price, price)
            # update maximum profit if selling today
            max_profit = max(max_profit, price - min_price)
        return max_profit

