"""
Federated Learning Simulator with Gradient Compression.

Core components:
  - FLServer: Maintains global model, aggregates compressed updates.
  - FLClient: Performs local training, compresses gradients before sending.
  - fl_train: Orchestrates the federated training loop.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Callable, Tuple, Optional

from compressors import get_compressor


# =============================================================================
# Server
# =============================================================================

class FLServer:
    """
    Central server for federated learning.

    Responsibilities:
      - Maintain and distribute the global model
      - Aggregate client gradient updates (FedAvg style)
      - Evaluate global model on test set
    """

    def __init__(self, model: nn.Module, lr: float = 0.01, device: str = "cpu"):
        self.global_model = model.to(device)
        self.lr = lr
        self.device = device

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return a copy of global model parameters."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]],
                  client_weights: Optional[List[float]] = None):
        """
        Aggregate client gradient updates into the global model.

        Uses weighted averaging (FedAvg). If client_weights is None,
        uses uniform weighting.

        Handles both:
          - state_dict keys (from fedavg mode)
          - named_parameters keys (from raw gradient mode)

        Args:
            client_updates: List of dicts mapping param_name -> gradient tensor.
            client_weights:  Optional weights for each client (e.g., by dataset size).
        """
        if not client_updates:
            return

        num_clients = len(client_updates)

        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        # Determine which keys the updates use
        update_keys = list(client_updates[0].keys())

        # Apply updates via named_parameters (works for both key formats)
        # For raw gradients: keys match named_parameters (e.g., "conv1.weight")
        # For fedavg deltas: keys match state_dict (same names for trainable params)
        with torch.no_grad():
            param_dict = dict(self.global_model.named_parameters())

            for key in update_keys:
                if key not in param_dict:
                    # Skip non-trainable parameters (e.g., batch norm running stats)
                    continue

                aggregated_grad = torch.zeros_like(param_dict[key], dtype=torch.float32)
                for i, update in enumerate(client_updates):
                    aggregated_grad += client_weights[i] * update[key].float()

                # Apply: w = w - lr * aggregated_gradient
                param_dict[key].data = (
                    param_dict[key].data.float() - self.lr * aggregated_grad
                ).to(param_dict[key].dtype)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate global model on test data.

        Returns:
            (test_loss, test_accuracy)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                total_loss += criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)

        return total_loss / total, correct / total


# =============================================================================
# Client
# =============================================================================

class FLClient:
    """
    Federated learning client.

    Supports two gradient strategies:
      - "raw":    Compute gradient on one mini-batch, do NOT update local model.
                  Sends true gradient to server. (Distributed SGD)
      - "fedavg": Train locally for E epochs, then send the pseudo-gradient
                  delta = w_global - w_local. (McMahan et al. 2017)

    The gradient strategy affects what the compressor receives:
      - "raw" gradients are noisier but statistically well-characterized.
      - "fedavg" deltas are smoother but accumulate multiple update steps.
    """

    def __init__(self, client_id: int, local_data: DataLoader,
                 model_fn: Callable, compress_fn: Callable,
                 bits_fn: Callable, device: str = "cpu"):
        """
        Args:
            client_id:   Unique client identifier.
            local_data:  DataLoader for this client's local dataset.
            model_fn:    Callable that returns a fresh model instance.
            compress_fn: Gradient compression function.
            bits_fn:     Function to estimate bits for a tensor.
            device:      Compute device.
        """
        self.client_id = client_id
        self.local_data = local_data
        self.model = model_fn().to(device)
        self.compress_fn = compress_fn
        self.bits_fn = bits_fn
        self.device = device
        self.num_samples = len(local_data.dataset)

        # Iterator for batch-wise gradient mode
        self._data_iter = None

    def _get_next_batch(self):
        """Get next mini-batch, resetting iterator if exhausted."""
        if self._data_iter is None:
            self._data_iter = iter(self.local_data)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.local_data)
            return next(self._data_iter)

    def compute_raw_gradient(self, global_weights: Dict[str, torch.Tensor],
                              ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Compute raw gradient on ONE mini-batch without updating local model.

        This matches the TF simulator behavior:
          1. Load global weights
          2. Forward + backward on one batch
          3. Extract gradients (do NOT apply them)
          4. Compress and return

        Returns:
            (compressed_gradient_dict, total_bits_transmitted)
        """
        # Load global weights (no local updates will be made)
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()

        # Get one mini-batch
        data, target = self._get_next_batch()
        data, target = data.to(self.device), target.to(self.device)

        # Forward + backward (compute gradients only)
        self.model.zero_grad()
        output = self.model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()

        # Extract, compress, and package gradients
        gradient_update = {}
        total_bits = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().clone()

                # Compress the gradient
                compressed_grad = self.compress_fn(grad)
                gradient_update[name] = compressed_grad

                # Track communication cost
                total_bits += self.bits_fn(grad)

        return gradient_update, total_bits

    def compute_fedavg_update(self, global_weights: Dict[str, torch.Tensor],
                               local_epochs: int = 1, local_lr: float = 0.01,
                               ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        FedAvg: train locally for E epochs, return compressed pseudo-gradient.

        The pseudo-gradient is: delta = w_global - w_local_after_training
        This is NOT a true gradient but an accumulated weight difference.

        Returns:
            (compressed_gradient_dict, total_bits_transmitted)
        """
        # Load global weights
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=local_lr)
        criterion = nn.CrossEntropyLoss()

        # Local training (multiple epochs, model IS updated)
        for epoch in range(local_epochs):
            for data, target in self.local_data:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Compute pseudo-gradient: delta = w_global - w_local
        local_weights = self.model.state_dict()
        gradient_update = {}
        total_bits = 0

        # Map state_dict keys to parameter names for consistency
        # state_dict uses keys like "conv1.weight", same as named_parameters
        for key in global_weights:
            delta = global_weights[key].float() - local_weights[key].float()

            compressed_delta = self.compress_fn(delta)
            gradient_update[key] = compressed_delta

            total_bits += self.bits_fn(delta)

        return gradient_update, total_bits


# =============================================================================
# Experimental Unit (orchestrates experiments)
# =============================================================================

class ExperimentalUnit:
    """
    Orchestrates federated learning experiments.

    Manages the interaction between server and clients across
    communication rounds, collects metrics, and supports
    different gradient strategies.
    """

    def __init__(self, server: FLServer, clients: List[FLClient],
                 test_loader: DataLoader):
        self.server = server
        self.clients = clients
        self.test_loader = test_loader

    def run(self, num_rounds: int, gradient_strategy: str = "raw",
            local_epochs: int = 1, local_lr: float = 0.01,
            eval_every: int = 1, verbose: bool = True) -> Dict:
        """
        Execute the federated training loop.

        Args:
            num_rounds:        Number of communication rounds.
            gradient_strategy: "raw" (one-batch gradient, no local updates)
                               or "fedavg" (local training + pseudo-gradient).
            local_epochs:      Epochs per round (only used in "fedavg" mode).
            local_lr:          Local learning rate.
            eval_every:        Evaluate every N rounds.
            verbose:           Print progress.

        Returns:
            History dict with metrics per round.
        """
        history = {
            "accuracy": [], "loss": [],
            "bits_per_round": [], "cumulative_bits": [],
            "rounds": [],
        }
        total_bits = 0

        # Initial evaluation
        init_loss, init_acc = self.server.evaluate(self.test_loader)
        if verbose:
            print(f"[Round 0] Loss: {init_loss:.4f} | Accuracy: {init_acc:.4f}")
        history["rounds"].append(0)
        history["accuracy"].append(init_acc)
        history["loss"].append(init_loss)
        history["bits_per_round"].append(0)
        history["cumulative_bits"].append(0)

        for round_num in range(1, num_rounds + 1):
            global_weights = self.server.get_global_weights()

            # Collect updates from all clients
            client_updates = []
            client_weights = []
            round_bits = 0

            for client in self.clients:
                if gradient_strategy == "raw":
                    update, bits = client.compute_raw_gradient(global_weights)
                elif gradient_strategy == "fedavg":
                    update, bits = client.compute_fedavg_update(
                        global_weights,
                        local_epochs=local_epochs,
                        local_lr=local_lr,
                    )
                else:
                    raise ValueError(
                        f"Unknown gradient_strategy: '{gradient_strategy}'. "
                        f"Use 'raw' or 'fedavg'."
                    )

                client_updates.append(update)
                client_weights.append(client.num_samples)
                round_bits += bits

            total_bits += round_bits

            # Aggregate updates at server
            self.server.aggregate(client_updates, client_weights)

            # Evaluate
            if round_num % eval_every == 0 or round_num == num_rounds:
                test_loss, test_acc = self.server.evaluate(self.test_loader)
                history["rounds"].append(round_num)
                history["accuracy"].append(test_acc)
                history["loss"].append(test_loss)
                history["bits_per_round"].append(round_bits)
                history["cumulative_bits"].append(total_bits)

                if verbose:
                    bits_mb = total_bits / (8 * 1024 * 1024)
                    print(
                        f"[Round {round_num:3d}] "
                        f"Loss: {test_loss:.4f} | "
                        f"Acc: {test_acc:.4f} | "
                        f"Round bits: {round_bits:,} | "
                        f"Total: {bits_mb:.2f} MB"
                    )

        return history


# =============================================================================
# Convenience function (wraps everything)
# =============================================================================

def fl_train(
    # Core FL parameters
    num_clients: int = 10,
    num_rounds: int = 50,
    local_epochs: int = 1,
    local_lr: float = 0.01,
    global_lr: float = 1.0,
    batch_size: int = 32,
    gradient_strategy: str = "raw",

    # Data parameters
    dataset_name: str = "mnist",
    partition_method: str = "iid",
    dirichlet_alpha: float = 0.5,
    max_samples: int = None,

    # Compression parameters
    compression_method: str = "none",
    compression_kwargs: dict = None,

    # System parameters
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
    eval_every: int = 1,
) -> Dict:
    """
    Run a complete federated learning experiment.

    Args:
        num_clients:        Number of federated clients.
        num_rounds:         Number of communication rounds.
        local_epochs:       Local training epochs per round (only for "fedavg").
        local_lr:           Learning rate for local SGD.
        global_lr:          Server learning rate (scales aggregated update).
        batch_size:         Mini-batch size for local training.
        gradient_strategy:  "raw" = compute gradient on one batch, no local updates.
                            "fedavg" = local training for E epochs, send delta.
        dataset_name:       Dataset to use ("mnist").
        partition_method:   Data split method ("iid" or "dirichlet").
        dirichlet_alpha:    Dirichlet parameter for non-IID splits.
        max_samples:        Cap on total training samples (None = use all).
        compression_method: Compression algorithm ("none", "signsgd", "qsgd", "mpgbp").
        compression_kwargs: Extra arguments for the compressor.
        device:             Compute device ("cpu" or "cuda").
        seed:               Random seed.
        verbose:            Print progress.
        eval_every:         Evaluate global model every N rounds.

    Returns:
        Dictionary with training history:
          - "accuracy": list of test accuracies per eval round
          - "loss": list of test losses per eval round
          - "bits": list of cumulative bits transmitted per round
          - "rounds": list of round numbers where eval happened
          - "config": experiment configuration dict
    """
    torch.manual_seed(seed)

    if compression_kwargs is None:
        compression_kwargs = {}

    # ---- Load data ----
    from data import load_mnist, partition_data, get_client_loader, get_test_loader
    from models import get_model

    if dataset_name == "mnist":
        train_dataset, test_dataset = load_mnist()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Optional: limit total samples
    if max_samples is not None and max_samples < len(train_dataset):
        from torch.utils.data import Subset
        indices = torch.randperm(len(train_dataset))[:max_samples].tolist()
        train_dataset = Subset(train_dataset, indices)

    # Partition data across clients
    client_datasets = partition_data(
        train_dataset, num_clients,
        method=partition_method, alpha=dirichlet_alpha, seed=seed
    )

    test_loader = get_test_loader(test_dataset)

    # ---- Initialize compressor ----
    compress_fn, bits_fn = get_compressor(compression_method, **compression_kwargs)

    # ---- Initialize server ----
    model_fn = lambda: get_model(dataset_name)
    server = FLServer(model=model_fn(), lr=global_lr, device=device)

    # ---- Initialize clients ----
    clients = []
    for i in range(num_clients):
        client_loader = get_client_loader(client_datasets[i], batch_size)
        client = FLClient(
            client_id=i,
            local_data=client_loader,
            model_fn=model_fn,
            compress_fn=compress_fn,
            bits_fn=bits_fn,
            device=device,
        )
        clients.append(client)

    # ---- Run via ExperimentalUnit ----
    experiment = ExperimentalUnit(server, clients, test_loader)
    history = experiment.run(
        num_rounds=num_rounds,
        gradient_strategy=gradient_strategy,
        local_epochs=local_epochs,
        local_lr=local_lr,
        eval_every=eval_every,
        verbose=verbose,
    )

    # Attach config to history
    history["config"] = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "local_lr": local_lr,
        "global_lr": global_lr,
        "batch_size": batch_size,
        "gradient_strategy": gradient_strategy,
        "compression": compression_method,
        "compression_kwargs": compression_kwargs,
        "partition": partition_method,
        "dirichlet_alpha": dirichlet_alpha,
    }

    return history
