"""
dataset.py — MNIST data loading and federated partitioning.

Supports two partitioning modes:
  • IID      — each client receives a uniformly random subset
  • Non-IID  — labels are distributed via a Dirichlet distribution,
               creating realistic heterogeneous data across clients
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config


# ── Standard MNIST transform ──────────────────────────────────────────
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean/std
])


def _load_mnist():
    """Download (if needed) and return the full MNIST train & test sets."""
    train_ds = datasets.MNIST(
        root=config.DATA_DIR, train=True, download=True, transform=_transform
    )
    test_ds = datasets.MNIST(
        root=config.DATA_DIR, train=False, download=True, transform=_transform
    )
    return train_ds, test_ds


# ── Partitioning strategies ───────────────────────────────────────────

def partition_iid(dataset, num_clients: int):
    """
    Split *dataset* into *num_clients* equal-sized random subsets.

    Returns
    -------
    client_indices : list[np.ndarray]
        Per-client index arrays into the original dataset.
    """
    total = len(dataset)
    indices = np.random.permutation(total)
    splits = np.array_split(indices, num_clients)
    return splits


def partition_noniid(dataset, num_clients: int, alpha: float = config.DIRICHLET_ALPHA):
    """
    Partition *dataset* across *num_clients* using a Dirichlet distribution
    over labels, controlled by concentration parameter *alpha*.

    Lower alpha → more heterogeneous (skewed) distributions.

    Returns
    -------
    client_indices : list[np.ndarray]
        Per-client index arrays into the original dataset.
    """
    targets = np.array(list(dataset.targets))
    num_classes = len(np.unique(targets))

    # Draw a Dirichlet proportion vector for each class
    client_indices = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_idxs = np.where(targets == cls)[0]
        np.random.shuffle(cls_idxs)

        # Dirichlet proportions for this class across clients
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Convert proportions to actual counts
        proportions = (proportions * len(cls_idxs)).astype(int)
        # Fix rounding so total matches
        proportions[-1] = len(cls_idxs) - proportions[:-1].sum()

        splits = np.split(cls_idxs, np.cumsum(proportions)[:-1])
        for client_id, split in enumerate(splits):
            client_indices[client_id].append(split)

    # Concatenate all class-level indices per client
    client_indices = [np.concatenate(idxs) for idxs in client_indices]
    return client_indices


# ── Public API ─────────────────────────────────────────────────────────

def get_partitioned_data(partition: str = config.DATA_PARTITION,
                         num_clients: int = config.NUM_CLIENTS):
    """
    Return partitioned training data and a shared test set.

    Parameters
    ----------
    partition : str
        "iid" or "noniid"
    num_clients : int
        Number of FL clients to partition for.

    Returns
    -------
    client_train_indices : list[np.ndarray]
    train_dataset : torchvision.datasets.MNIST
    test_dataset  : torchvision.datasets.MNIST
    """
    train_ds, test_ds = _load_mnist()

    if partition == "iid":
        client_indices = partition_iid(train_ds, num_clients)
    elif partition == "noniid":
        client_indices = partition_noniid(train_ds, num_clients)
    else:
        raise ValueError(f"Unknown partition mode: {partition!r}")

    return client_indices, train_ds, test_ds


def get_client_loaders(client_id: int,
                       client_indices,
                       train_dataset,
                       test_dataset,
                       batch_size: int = config.BATCH_SIZE):
    """
    Build DataLoaders for a specific client.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """
    train_subset = Subset(train_dataset, client_indices[client_id])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_client_labels(client_id: int, client_indices, train_dataset):
    """Return the label array for a specific client's training subset."""
    indices = client_indices[client_id]
    targets = np.array(list(train_dataset.targets))
    return targets[indices]
