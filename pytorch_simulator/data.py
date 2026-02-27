# -*- coding: utf-8 -*-
"""
Data loading and partitioning utilities for FL experiments.

Supports:
  - IID partitioning (uniform random split)
  - Non-IID partitioning via Dirichlet distribution
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_mnist(data_dir: str = "./data"):
    """
    Load MNIST train and test datasets.

    Returns:
        (train_dataset, test_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def partition_iid(dataset, num_clients: int, seed: int = 42):
    """
    Partition dataset into IID subsets for each client.

    Args:
        dataset:     Full training dataset.
        num_clients: Number of federated clients.
        seed:        Random seed for reproducibility.

    Returns:
        List of Subset objects, one per client.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, s.tolist()) for s in splits]


def partition_noniid_dirichlet(dataset, num_clients: int, alpha: float = 0.5,
                                seed: int = 42):
    """
    Partition dataset into non-IID subsets using Dirichlet distribution.

    Lower alpha = more heterogeneous (each client gets fewer classes).
    alpha -> inf = IID.

    Args:
        dataset:     Full training dataset with .targets attribute.
        num_clients: Number of federated clients.
        alpha:       Dirichlet concentration parameter.
        seed:        Random seed.

    Returns:
        List of Subset objects, one per client.
    """
    rng = np.random.default_rng(seed)

    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Compute split sizes
        proportions = proportions / proportions.sum()
        split_sizes = (proportions * len(class_indices)).astype(int)

        # Fix rounding errors
        diff = len(class_indices) - split_sizes.sum()
        for i in range(abs(diff)):
            idx = i % num_clients
            split_sizes[idx] += 1 if diff > 0 else -1

        # Distribute indices
        start = 0
        for client_id in range(num_clients):
            end = start + split_sizes[client_id]
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end

    return [Subset(dataset, indices) for indices in client_indices]


def partition_data(dataset, num_clients: int, method: str = "iid",
                   alpha: float = 0.5, seed: int = 42):
    """
    Unified data partitioning interface.

    Args:
        dataset:     Training dataset.
        num_clients: Number of clients.
        method:      "iid" or "dirichlet".
        alpha:       Dirichlet parameter (only used if method="dirichlet").
        seed:        Random seed.

    Returns:
        List of client data subsets.
    """
    if method == "iid":
        return partition_iid(dataset, num_clients, seed)
    elif method == "dirichlet":
        return partition_noniid_dirichlet(dataset, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown partitioning method: {method}")


def get_client_loader(client_data, batch_size: int, shuffle: bool = True):
    """Create a DataLoader for a client's local dataset."""
    return DataLoader(client_data, batch_size=batch_size, shuffle=shuffle)


def get_test_loader(test_dataset, batch_size: int = 256):
    """Create a DataLoader for the global test set."""
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
