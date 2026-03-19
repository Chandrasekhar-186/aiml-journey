import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Get config from environment
model_name = os.getenv('MODEL_NAME', 'RandomForest')
threshold = float(os.getenv('ACCURACY_THRESHOLD', 85))

# Train model
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test) * 100

print(f"Model: {model_name}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Passed threshold: {accuracy >= threshold}")
