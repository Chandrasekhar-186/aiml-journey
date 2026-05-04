# Phase 2 Day 22 — LeetCode #714
# Stock with Transaction Fee
# Difficulty: Medium ⚡
# Approach: DP state machine
# Time: O(n) | Space: O(1)

def maxProfit(prices, fee):
    # held: max profit while holding stock
    # cash: max profit while NOT holding
    held = -prices[0]
    cash = 0

    for price in prices[1:]:
        held = max(held, cash - price)
        cash = max(cash, held + price - fee)

    return cash

print(maxProfit([1,3,2,8,4,9], 2))  # 8
print(maxProfit([1,3,7,5,10,3], 3)) # 6

# Phase 2 Day 22 — LeetCode #901
# Online Stock Span
# Difficulty: Medium ⚡
# Approach: Monotonic stack
# Time: O(n) | Space: O(n)

class StockSpanner:
    def __init__(self):
        self.stack = []  # (price, span)

    def next(self, price):
        span = 1
        while (self.stack and
               self.stack[-1][0] <= price):
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span

spanner = StockSpanner()
print(spanner.next(100))  # 1
print(spanner.next(80))   # 1
print(spanner.next(60))   # 1
print(spanner.next(70))   # 2
print(spanner.next(60))   # 1
print(spanner.next(75))   # 4
print(spanner.next(85))   # 6
