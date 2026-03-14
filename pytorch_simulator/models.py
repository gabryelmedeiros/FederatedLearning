"""
Model architectures for FL experiments.

Currently supports:
  - MNISTNet:   CNN matching the SBrT 2025 paper architecture (~591K params)
  - CIFAR10Net: Small 3-conv CNN for CIFAR-10 (~132K params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    CNN for MNIST digit classification.

    Architecture matches the SBrT 2025 paper:
      Conv2d(1, 32, 3x3, ReLU) -> MaxPool(3x3, stride=2) -> Flatten ->
      Linear(4608, 128, ReLU) -> Linear(128, 10, Softmax)

    Total parameters: 591,562
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # 28x28 -> 26x26
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)          # 26x26 -> 12x12
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (B, 32, 26, 26)
        x = self.pool(x)                 # (B, 32, 12, 12)
        x = torch.flatten(x, 1)          # (B, 4608)
        x = F.relu(self.fc1(x))          # (B, 128)
        x = self.fc2(x)                  # (B, 10)
        return x


class CIFAR10Net(nn.Module):
    """
    Small CNN for CIFAR-10 classification.

    Architecture:
      Conv2d(3, 32, 3, pad=1) -> ReLU -> MaxPool(2,2)   # 32x16x16
      Conv2d(32, 64, 3, pad=1) -> ReLU -> MaxPool(2,2)  # 64x8x8
      Conv2d(64, 64, 3, pad=1) -> ReLU -> MaxPool(2,2)  # 64x4x4
      Flatten                                             # 1024
      Linear(1024, 128) -> ReLU
      Linear(128, 10)

    Total parameters: ~132K, more evenly distributed across layers than MNISTNet.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 4 * 4, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 64, 4, 4)
        x = torch.flatten(x, 1)               # (B, 1024)
        x = F.relu(self.fc1(x))               # (B, 128)
        x = self.fc2(x)                        # (B, 10)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str = "mnist") -> nn.Module:
    """Factory for model creation."""
    if name == "mnist":
        return MNISTNet()
    elif name == "cifar10":
        return CIFAR10Net()
    else:
        raise ValueError(f"Unknown model: {name}")
