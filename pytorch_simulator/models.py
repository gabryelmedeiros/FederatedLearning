# -*- coding: utf-8 -*-
"""
Model architectures for FL experiments.

Currently supports:
  - MNISTNet: CNN matching the SBrT 2025 paper architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    CNN for MNIST digit classification.

    Architecture matches the SBrT 2025 paper:
      Conv2d(1, 32, 3x3, ReLU) -> MaxPool(3x3, stride=2) -> Flatten ->
      Linear(5408, 128, ReLU) -> Linear(128, 10, Softmax)

    Total parameters: ~693K
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


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str = "mnist") -> nn.Module:
    """Factory for model creation."""
    if name == "mnist":
        return MNISTNet()
    else:
        raise ValueError(f"Unknown model: {name}")
