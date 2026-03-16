# Day 04 — Exception Handling + File I/O
# Date: March 16, 2026

import json
import os

# 1. Basic exception handling
def load_model_config(filepath):
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in: {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        print("Config load attempt complete")

# 2. Custom exception
class ModelAccuracyError(Exception):
    def __init__(self, accuracy, threshold):
        self.accuracy = accuracy
        self.threshold = threshold
        super().__init__(
            f"Accuracy {accuracy}% below threshold {threshold}%"
        )

def validate_model(accuracy, threshold=80):
    if accuracy < threshold:
        raise ModelAccuracyError(accuracy, threshold)
    return True

# 3. Writing results to file
def save_experiment_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")

# Test it
results = {
    "model": "RandomForest",
    "accuracy": 92.5,
    "params": {"n_estimators": 100, "max_depth": 5}
}
save_experiment_results(results, "experiment_01.json")

try:
    validate_model(75)
except ModelAccuracyError as e:
    print(f"Validation failed: {e}")
