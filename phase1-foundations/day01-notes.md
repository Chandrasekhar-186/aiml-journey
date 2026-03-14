# Day 1 Notes 
Git — Key Commands
git init                          # start a repo
git clone [url]                   # copy repo to local
git status                        # see what changed
git add .                         # stage all changes
git commit -m "message"           # save snapshot
git push origin main              # upload to GitHub
git pull                          # download latest changes
git log --oneline                 # see commit history


Big-O Notation
Notation	Name	Plain English	Example
O(1)	Constant	Always takes same time, no matter input size	Accessing array index
O(log n)	Logarithmic	Cuts problem in half each step — very fast	Binary search
O(n)	Linear	One operation per input item	Loop through a list
O(n²)	Quadratic	Nested loops — slows down fast	Bubble sort, nested for loops
Real-world analogy: O(1) = light switch. O(n) = reading every page of a book. O(n²) = comparing every page with every other page. O(log n) = opening book to middle, then half again (binary search).
