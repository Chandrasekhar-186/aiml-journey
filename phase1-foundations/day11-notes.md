## PyTorch Core Concepts

Tensor    = PyTorch's ndarray (like numpy + GPU support)
Autograd  = automatic differentiation engine
nn.Module = base class for all neural networks
optimizer = updates weights (Adam, SGD, RMSprop)
loss      = measures prediction error

## Training Loop — Always same structure:
for epoch in range(epochs):
    optimizer.zero_grad()  # clear gradients
    output = model(X)      # forward pass
    loss = criterion(output, y)  # compute loss
    loss.backward()        # backward pass
    optimizer.step()       # update weights

## MLflow Model Registry — Stages
None → Staging → Production → Archived
Use staging for testing
Use production for serving
Never delete — archive instead!

## BFS Template
from collections import deque
queue = deque([start])
visited = set([start])
while queue:
    node = queue.popleft()
    for neighbor in get_neighbors(node):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

## DFS vs BFS
DFS: uses stack/recursion, explores deep first
     → islands, trees, connected components
BFS: uses queue, explores wide first
     → shortest path, level order, spreading
