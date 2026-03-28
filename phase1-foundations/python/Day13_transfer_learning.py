# Day 13 — Transfer Learning with ResNet
# Date: March 25, 2026
# Real CV engineers use pretrained models!

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import mlflow
import mlflow.pytorch

device = torch.device('cuda' if
                       torch.cuda.is_available()
                       else 'cpu')

# 1. Load pretrained ResNet18
# Trained on ImageNet (1.2M images, 1000 classes)
resnet = models.resnet18(pretrained=True)

print(f"ResNet18 architecture:")
print(f"Total params: "
      f"{sum(p.numel() for p in resnet.parameters()):,}")

# 2. Freeze all layers — keep pretrained weights!
for param in resnet.parameters():
    param.requires_grad = False

# 3. Replace final layer for CIFAR-10 (10 classes)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)
# Only final layer trains — much faster!

trainable = sum(p.numel() for p in
                resnet.parameters()
                if p.requires_grad)
print(f"Trainable params: {trainable:,}")
# Much less than training from scratch!

resnet = resnet.to(device)

# 4. Data with ImageNet normalization
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet needs 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet stats!
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)
# Use subset for speed
subset = torch.utils.data.Subset(
    trainset, range(2000)
)
trainloader = torch.utils.data.DataLoader(
    subset, batch_size=32, shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)
testsubset = torch.utils.data.Subset(
    testset, range(500)
)
testloader = torch.utils.data.DataLoader(
    testsubset, batch_size=32
)

# 5. Train only final layer — log to MLflow!
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    resnet.fc.parameters(), lr=0.001
)

mlflow.set_experiment("transfer_learning_resnet")

with mlflow.start_run(run_name="ResNet18_finetune"):
    mlflow.log_param("base_model", "ResNet18")
    mlflow.log_param("pretrained", True)
    mlflow.log_param("frozen_layers", "all_except_fc")
    mlflow.log_param("dataset", "CIFAR10_subset")
    mlflow.log_param("epochs", 3)

    for epoch in range(3):
        resnet.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate
        resnet.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                _, predicted = torch.max(
                    resnet(inputs), 1
                )
                total += labels.size(0)
                correct += (predicted ==
                             labels).sum().item()

        acc = correct / total
        avg_loss = running_loss / len(trainloader)
        mlflow.log_metric("accuracy", acc,
                           step=epoch)
        mlflow.log_metric("loss", avg_loss,
                           step=epoch)
        print(f"Epoch {epoch+1}: "
              f"Loss={avg_loss:.4f} "
              f"Acc={acc:.4f}")

    mlflow.pytorch.log_model(resnet,
                              "resnet18_finetuned")
    print("Transfer learning model logged!")

# 6. Compare: scratch vs transfer learning
print("\nKey insight:")
print("SimpleCNN from scratch: ~50-60% accuracy")
print("ResNet18 transfer:      ~70-80% accuracy")
print("Same dataset, same epochs — huge difference!")
