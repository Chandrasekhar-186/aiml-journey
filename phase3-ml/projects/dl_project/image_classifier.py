# Phase 3 DL Project — Image Classification
# Date: May 21, 2026
# Complete DL pipeline: CNN + Attention + MLflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
import math
import time

print("="*60)
print("DL Capstone — Image Classifier")
print("with ResNet + Attention + MLflow")
print("="*60)

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)
print(f"Device: {device}")

# 1. Data pipeline with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test
)

train_loader = DataLoader(
    train_set, batch_size=128,
    shuffle=True, num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=128,
    shuffle=False, num_workers=2,
    pin_memory=True
)

classes = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')

# 2. Model: ResNet + Channel Attention (CBAM!)
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels,
                       channels // reduction,
                       bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction,
                       channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.fc(
            self.avg_pool(x).view(B, C)
        )
        mx = self.fc(
            self.max_pool(x).view(B, C)
        )
        scale = self.sigmoid(
            avg + mx
        ).view(B, C, 1, 1)
        return x * scale  # recalibrate!

class AttentionResBlock(nn.Module):
    """ResBlock with Channel Attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels,
            3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)  # attention!
        out += residual     # skip connection!
        return F.relu(out)

class AttentionResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3,
                       padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            AttentionResBlock(64),
            AttentionResBlock(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3,
                       stride=2, padding=1,
                       bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            AttentionResBlock(128),
            AttentionResBlock(128)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3,
                       stride=2, padding=1,
                       bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            AttentionResBlock(256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

model = AttentionResNet().to(device)
total_params = sum(
    p.numel() for p in model.parameters()
)
print(f"\nModel: AttentionResNet")
print(f"Parameters: {total_params:,}")

# 3. Training setup
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001, weight_decay=0.01
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=20,
    steps_per_epoch=len(train_loader)
)
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1  # modern trick!
)

# 4. Training loop
def train_epoch(model, loader,
                 optimizer, criterion,
                 scheduler):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        # Gradient clipping!
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total

def eval_model(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for i in range(len(y)):
                label = y[i].item()
                class_correct[label] += \
                    (pred[i] == y[i]).item()
                class_total[label] += 1
    per_class = {
        classes[i]:
            class_correct[i]/class_total[i]
        for i in range(10)
    }
    return (total_loss/len(loader),
            correct/total, per_class)

# 5. Full training with MLflow
mlflow.set_experiment("phase3_dl_project")
with mlflow.start_run(
        run_name="AttentionResNet_CIFAR10"):
    mlflow.log_params({
        "model": "AttentionResNet",
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "epochs": 20,
        "batch_size": 128,
        "label_smoothing": 0.1,
        "weight_decay": 0.01,
        "gradient_clipping": 1.0,
        "total_params": total_params,
        "attention": "ChannelAttention(SE)"
    })

    best_val_acc = 0
    print(f"\n{'Ep':>3} {'TrLoss':>8}"
          f"{'TrAcc':>8} {'VlAcc':>8}"
          f"{'Time':>7}")

    for epoch in range(20):
        start = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader,
            optimizer, criterion, scheduler
        )
        val_loss, val_acc, per_class = \
            eval_model(model, test_loader,
                        criterion)
        elapsed = time.time() - start

        mlflow.log_metrics({
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                "best_model.pth"
            )
            mlflow.log_artifact("best_model.pth")

        print(f"{epoch+1:>3} {tr_loss:>8.4f}"
              f"{tr_acc:>8.4f} {val_acc:>8.4f}"
              f"{elapsed:>6.1f}s")

    mlflow.log_metric("best_val_acc",
                       best_val_acc)

    # Per-class accuracy
    print("\nPer-class accuracy (best model):")
    for cls, acc in sorted(
            per_class.items(),
            key=lambda x: x[1],
            reverse=True
    ):
        bar = "█" * int(acc * 20)
        print(f"  {cls:10}: {acc:.3f} {bar}")

    print(f"\nBest validation accuracy: "
          f"{best_val_acc:.4f}")
    print("Model saved to MLflow! ✅")

print("\n" + "="*60)
print("DL Project COMPLETE! 🏆")
print("TOMORROW: COMPUTER VISION WEEK! 🎨")
print("="*60)
