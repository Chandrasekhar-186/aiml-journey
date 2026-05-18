# Phase 3 Day 10 — CNN Advanced Internals
# Date: May 18, 2026
# ResNet + skip connections + modern CNNs!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import mlflow

print("="*60)
print("CNN Advanced — ResNet + Modern Architectures")
print("="*60)

"""
CNN INTERNALS — COMPLETE MATH

Convolution operation:
(f * g)[n] = Σ f[m] * g[n-m]

For 2D images:
Output[i,j] = Σₖ Σₗ Input[i+k, j+l] * Kernel[k,l]

Output size:
H_out = (H_in - K + 2P) / S + 1
W_out = (W_in - K + 2P) / S + 1

Where:
  K = kernel size
  P = padding
  S = stride

Key properties:
→ Parameter sharing: same kernel across image
  → translational invariance!
→ Local connectivity: each neuron sees local region
→ Hierarchical features:
  Layer 1: edges, colors
  Layer 2: textures, shapes
  Layer 3: parts (eyes, wheels)
  Layer 4: objects (faces, cars)

POOLING:
Max pooling: takes maximum in region
  → Spatial invariance!
  → Reduces spatial dimensions

Average pooling: takes average
  → Smoother downsampling
  → Used in global avg pooling

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DEEP NETWORK PROBLEM:

Adding more layers should help.
But in practice: degradation problem!
→ Deeper networks perform WORSE than shallow!
→ Not overfitting — training error also worse
→ Optimization problem: gradients vanish

WHY? Stacking many layers:
→ Gradients shrink at each layer
→ Early layers learn essentially nothing
→ 50+ layers: practically no learning!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESNET — RESIDUAL LEARNING:

Key insight: instead of learning H(x),
             learn the RESIDUAL F(x) = H(x) - x

Regular block:    y = F(x)
Residual block:   y = F(x) + x  ← skip connection!

WHY THIS WORKS:
If optimal mapping is identity (H(x)=x):
→ F(x) just needs to learn 0 (easier!)
→ Skip connection provides "highway" for gradients
→ Gradient flows DIRECTLY through skip connection:
  ∂L/∂x = ∂L/∂y * (∂F/∂x + 1)
  The "+1" prevents vanishing gradient!

ResNet-18 architecture:
Conv1 (7×7, 64, stride=2)
MaxPool (3×3, stride=2)
Layer1: 2 × ResidualBlock(64)
Layer2: 2 × ResidualBlock(128, stride=2)
Layer3: 2 × ResidualBlock(256, stride=2)
Layer4: 2 × ResidualBlock(512, stride=2)
AvgPool → Flatten → FC(num_classes)
"""

# 1. Residual Block from scratch
class ResidualBlock(nn.Module):
    def __init__(self, in_channels,
                  out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or \
                in_channels != out_channels:
            # Adjust dimensions for skip path
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add skip connection (residual!)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 2. Mini ResNet for CIFAR-10
class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Test architecture
model = MiniResNet(num_classes=10)
x_test = torch.randn(4, 3, 32, 32)
output = model(x_test)
print(f"Input:  {x_test.shape}")
print(f"Output: {output.shape}")
total_params = sum(
    p.numel() for p in model.parameters()
)
print(f"Total parameters: {total_params:,}")

# 3. CIFAR-10 training
print("\n=== TRAINING ON CIFAR-10 ===")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
    shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_set, batch_size=128,
    shuffle=False, num_workers=2
)

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)
print(f"Device: {device}")

model = MiniResNet().to(device)
optimizer = optim.Adam(
    model.parameters(), lr=0.001
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader,
                 optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total

def eval_model(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss/len(loader), correct/total

mlflow.set_experiment("phase3_resnet_cifar10")
with mlflow.start_run(
        run_name="MiniResNet_CIFAR10"):
    mlflow.log_param("model", "MiniResNet")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("epochs", 5)
    mlflow.log_param("params", total_params)

    print(f"\n{'Epoch':>6} {'Train Loss':>11}"
          f"{'Train Acc':>10} {'Val Acc':>9}")
    for epoch in range(5):
        tr_loss, tr_acc = train_epoch(
            model, train_loader,
            optimizer, criterion
        )
        val_loss, val_acc = eval_model(
            model, test_loader, criterion
        )
        scheduler.step()

        mlflow.log_metric(
            "train_loss", tr_loss, step=epoch
        )
        mlflow.log_metric(
            "train_acc", tr_acc, step=epoch
        )
        mlflow.log_metric(
            "val_acc", val_acc, step=epoch
        )
        print(f"{epoch+1:>6} {tr_loss:>11.4f}"
              f"{tr_acc:>10.4f} {val_acc:>9.4f}")

# 4. Modern CNN architectures overview
print("\n=== MODERN CNN ARCHITECTURES ===")
print("""
VGG (2014): very deep, simple 3×3 convs
  → Proved depth matters!
  → 138M params (heavy!)

ResNet (2015): skip connections
  → Solved degradation problem
  → 50/101/152 layer variants
  → Used everywhere as backbone!

DenseNet (2016): all layers connected!
  → Each layer receives ALL previous outputs
  → Feature reuse → fewer params
  → Good for medical imaging

EfficientNet (2019): compound scaling
  → Scale width+depth+resolution together
  → Best accuracy/compute tradeoff
  → EfficientNet-B0 to B7 variants

ConvNeXt (2022): CNN modernized!
  → Borrows design from Vision Transformers
  → Depthwise conv + inverted bottleneck
  → Competitive with ViT at same cost

Architecture design rules:
→ Use 3×3 convolutions (efficient!)
→ Use BatchNorm after every conv
→ Use ReLU/GELU activations
→ Use skip connections for depth
→ Global average pooling before FC
→ No fully connected middle layers
""")

print("\n" + "="*60)
print("CNN Advanced + ResNet — MASTERED! 🎯")
print("="*60)
