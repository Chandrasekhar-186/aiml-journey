## Week 2 — Key Concepts Summary

Linear Algebra → vectors, matrices, eigenvalues,
                 cosine similarity (used in RAG!)
Probability    → Bayes, distributions, CLT
Hypothesis     → p-value, A/B testing, t-test
MLflow         → tracking, registry, staging
PyTorch        → tensors, autograd, training loop
CNN            → conv, pool, flatten, classify
Transfer       → freeze layers, retrain head only
OpenCV         → preprocess images for CNN input
Spark Stream   → readStream, writeStream, trigger
DP pattern     → dp array, base case, recurrence
BFS/DFS        → queue/stack, visited set, 4-dirs
Sliding Window → left/right pointers, shrink left

## Two Pointers Template
left, right = 0, len(arr) - 1
while left < right:
    if condition_met:
        update_answer()
    if move_left_condition:
        left += 1
    else:
        right -= 1

## OpenCV → PyTorch Pipeline
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
tensor = transforms.ToTensor()(img)
output = model(tensor.unsqueeze(0))
