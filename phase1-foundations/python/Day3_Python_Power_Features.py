# Day 03 — Python Power Features
# Date: March 15, 2026

# 1. Generator — memory efficient iteration
def model_accuracy_generator(models):
    for model in models:
        yield model['name'], model['accuracy']

models = [
    {'name': 'RF', 'accuracy': 88},
    {'name': 'XGB', 'accuracy': 92},
    {'name': 'NN', 'accuracy': 95}
]

# Generator uses almost zero memory vs list
gen = model_accuracy_generator(models)
for name, acc in gen:
    print(f"{name}: {acc}%")

# 2. List comprehensions
accuracies = [m['accuracy'] for m in models]
high_performers = [m['name'] for m in models 
                   if m['accuracy'] > 90]

# 3. Dictionary comprehension
acc_dict = {m['name']: m['accuracy'] for m in models}

# 4. *args and **kwargs
def log_experiment(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Params: {kwargs}")

log_experiment("run_1", "run_2", 
                lr=0.001, epochs=10, batch_size=32)

# 5. zip and enumerate
names = ['RF', 'XGB', 'NN']
scores = [88, 92, 95]

for i, (name, score) in enumerate(zip(names, scores)):
    print(f"{i+1}. {name}: {score}%")
