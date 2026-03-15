"""
model.py — CNN architectures and training / evaluation routines.

Supported datasets
──────────────────
  • MNIST    — 1-channel 28×28 grayscale images (MNISTNet)
  • CIFAR-10 — 3-channel 32×32 colour images   (CIFAR10Net)

Use ``get_model(dataset)`` to obtain the correct architecture.

Loss  : CrossEntropyLoss
Optim : Adam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config


# ── Architectures ─────────────────────────────────────────────────────

class MNISTNet(nn.Module):
    """Simple 2-layer CNN for MNIST classification (1×28×28 input)."""

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


class CIFAR10Net(nn.Module):
    """Simple 2-layer CNN for CIFAR-10 classification (3×32×32 input)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two conv+pool layers: 32→16→8, so 64 * 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64,  8,  8)
        x = x.view(x.size(0), -1)              # (B, 64*8*8)
        x = F.relu(self.fc1(x))                # (B, 128)
        x = self.fc2(x)                        # (B, 10)
        return x


# ── Model factory ─────────────────────────────────────────────────────

def get_model(dataset: str = config.DATASET) -> nn.Module:
    """Return the appropriate CNN for the given dataset name."""
    if dataset == "mnist":
        return MNISTNet()
    elif dataset == "cifar10":
        return CIFAR10Net()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'mnist' or 'cifar10'.")


# ── Training & evaluation ─────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
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


def evaluate(model: nn.Module,
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
