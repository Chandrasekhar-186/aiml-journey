# Day 02 — Advanced OOP
# Date: March 14, 2026
# Concept: Magic methods, @property, Inheritance

class MLModel:
    model_count = 0

    def __init__(self, name, accuracy):
        self.name = name
        self.accuracy = accuracy
        MLModel.model_count += 1

    def __repr__(self):
        return f"MLModel('{self.name}', {self.accuracy})"

    def __str__(self):
        return f"{self.name} — Accuracy: {self.accuracy}%"

    @property
    def grade(self):
        if self.accuracy >= 90:
            return "Excellent"
        elif self.accuracy >= 75:
            return "Good"
        return "Needs Improvement"

    @classmethod
    def get_count(cls):
        return cls.model_count

class DeepLearningModel(MLModel):
    def __init__(self, name, accuracy, framework):
        super().__init__(name, accuracy)
        self.framework = framework

    def __str__(self):
        return f"{self.name} ({self.framework}) — Accuracy: {self.accuracy}%"

# Test it
m1 = MLModel("RandomForest", 88)
m2 = DeepLearningModel("ResNet50", 95, "PyTorch")
print(m1)
print(m2)
print(f"Grade: {m2.grade}")
print(f"Total models: {MLModel.get_count()}")
