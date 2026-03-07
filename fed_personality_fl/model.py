"""
model.py — CNN architecture and training / evaluation routines for MNIST.

Architecture
────────────
Conv2d(1 → 32, 3) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, 3) → ReLU → MaxPool2d(2)
Flatten
Linear(64 × 5 × 5 → 128) → ReLU
Linear(128 → 10)

Loss  : CrossEntropyLoss
Optim : Adam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config


class MNISTNet(nn.Module):
    """Simple 2-layer CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two conv+pool layers: 28→14→7, so 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64,  7,  7)
        x = x.view(x.size(0), -1)              # (B, 64*7*7)
        x = F.relu(self.fc1(x))                # (B, 128)
        x = self.fc2(x)                        # (B, 10)
        return x


def train_one_epoch(model: MNISTNet,
                    loader: DataLoader,
                    device: torch.device,
                    lr: float = config.LEARNING_RATE) -> float:
    """
    Train *model* for one epoch on *loader*.

    Returns
    -------
    avg_loss : float
        Mean training loss over all batches.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    num_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model: MNISTNet,
             loader: DataLoader,
             device: torch.device):
    """
    Evaluate *model* on *loader*.

    Returns
    -------
    loss     : float   — average cross-entropy loss
    accuracy : float   — fraction of correct predictions
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
