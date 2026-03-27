# Day 12 — CNN + Computer Vision Foundations
# Date: March 24, 2026
# Your CV journey officially begins! 🎯

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch

# 1. Load CIFAR-10 dataset (10 image classes)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32,
    shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32,
    shuffle=False, num_workers=0
)

classes = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')

# 2. Build CNN — your first vision model!
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv layers — extract features
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Fully connected layers — classify
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1: Conv → ReLU → Pool
        x = self.pool(self.relu(self.conv1(x)))
        # Block 2: Conv → ReLU → Pool
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten for FC layers
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if
                       torch.cuda.is_available()
                       else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model architecture:\n{model}")
print(f"Total parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# 3. Train for 3 epochs — log to MLflow!
mlflow.set_experiment("cifar10_cnn")

with mlflow.start_run(run_name="SimpleCNN_v1"):
    mlflow.log_param("architecture", "SimpleCNN")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("dataset", "CIFAR-10")

    for epoch in range(3):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(
                trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 200 == 199:
                avg_loss = running_loss / 200
                print(f"Epoch {epoch+1}, "
                      f"Batch {i+1}: "
                      f"Loss={avg_loss:.3f}")
                mlflow.log_metric(
                    "train_loss", avg_loss,
                    step=epoch * len(trainloader) + i
                )
                running_loss = 0.0

        # Evaluate each epoch
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(
                    outputs, 1)
                total += labels.size(0)
                correct += (predicted ==
                             labels).sum().item()

        acc = correct / total
        mlflow.log_metric("test_accuracy",
                           acc, step=epoch)
        print(f"Epoch {epoch+1} Accuracy: "
              f"{acc:.4f}")

    mlflow.pytorch.log_model(model, "cnn_model")
    print("CNN logged to MLflow!")

# 4. Class-wise accuracy
class_correct = [0] * 10
class_total = [0] * 10
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (
                predicted[i] == label).item()
            class_total[label] += 1

print("\nClass-wise accuracy:")
for i in range(10):
    acc = class_correct[i] / class_total[i]
    print(f"  {classes[i]:10s}: {acc:.2%}")
