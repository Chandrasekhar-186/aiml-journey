## Monotonic Deque Pattern
Maintains decreasing/increasing order
Front = current window maximum/minimum
Remove from front when outside window
Remove from back when smaller than new element

Template:
dq = deque()  # stores indices
for i, num in enumerate(nums):
    # Remove out-of-window from front
    while dq and dq[0] < i - k + 1:
        dq.popleft()
    # Maintain monotonic order from back
    while dq and nums[dq[-1]] < num:
        dq.pop()
    dq.append(i)
    # Window complete
    if i >= k - 1:
        result.append(nums[dq[0]])

## Anagram = Same Character Frequencies
Use Counter comparison: O(26) = O(1)
Sliding window: add new, remove old
Check equality at each valid window position

## Phase 1 Complete — Key Takeaways
1. Consistency > Intensity
   87 hours over 30 days > one 87-hour grind
2. Build in public
   28 LinkedIn posts = recruiter visibility
3. Use the tools you're targeting
   MLflow every day = interview confidence
4. Teach to learn
   Writing notes = 3× better retention
5. Start early on hard things
   System design Day 18 → 45 days ahead

## Phase 2 Mindset
Phase 1: breadth — cover everything
Phase 2: depth — master Spark completely
Goal: understand Spark like its creators do
Not just how to USE it — but WHY it works!
