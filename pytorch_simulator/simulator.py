# -*- coding: utf-8 -*-
"""
Federated Learning Simulator with Gradient Compression.

Two gradient strategies:
  - "raw": Per epoch: distribute weights, iterate over all batches,
           each batch: clients compute gradients -> server aggregates with Adam.
           Clients never update their own models.
  - "fedavg": Standard McMahan et al. 2017.
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
    def __init__(self, model: nn.Module, lr: float = 0.001,
                 optimizer_type: str = "adam", device: str = "cpu"):
        self.global_model = model.to(device)
        self.lr = lr
        self.device = device
        self.optimizer_type = optimizer_type

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.global_model.parameters(), lr=lr
            )
        else:
            self.optimizer = None

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_and_apply(self, client_gradients: List[List[torch.Tensor]]):
        """
        Average gradient lists from clients and apply to global model.
        Used in "raw" mode:
            average = mean(client_gradients)
            optimizer.apply_gradients(zip(average, model.trainable_weights))
        """
        if not client_gradients:
            return

        num_clients = len(client_gradients)
        params = list(self.global_model.parameters())

        # Average gradients across clients
        avg_grads = []
        for p_idx in range(len(params)):
            grad_sum = torch.zeros_like(params[p_idx], dtype=torch.float32)
            for c_idx in range(num_clients):
                grad_sum += client_gradients[c_idx][p_idx].float()
            avg_grads.append(grad_sum / num_clients)

        # Apply via optimizer
        if self.optimizer_type == "adam" and self.optimizer is not None:
            self.optimizer.zero_grad()
            for param, grad in zip(params, avg_grads):
                param.grad = grad.to(param.dtype)
            self.optimizer.step()
        else:
            with torch.no_grad():
                for param, grad in zip(params, avg_grads):
                    param.data -= self.lr * grad.to(param.dtype)

    def aggregate_dicts_and_apply(self, client_updates: List[Dict[str, torch.Tensor]],
                                   client_weights: Optional[List[float]] = None):
        """
        Aggregate dict-format updates (FedAvg mode).
        """
        if not client_updates:
            return

        num_clients = len(client_updates)
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        update_keys = list(client_updates[0].keys())
        param_dict = dict(self.global_model.named_parameters())

        with torch.no_grad():
            for key in update_keys:
                if key not in param_dict:
                    continue
                aggregated = torch.zeros_like(param_dict[key], dtype=torch.float32)
                for i, update in enumerate(client_updates):
                    aggregated += client_weights[i] * update[key].float()

                # FedAvg uses SGD: w = w - lr * delta
                param_dict[key].data = (
                    param_dict[key].data.float() - self.lr * aggregated
                ).to(param_dict[key].dtype)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
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
    def __init__(self, client_id: int, local_data: DataLoader,
                 model_fn: Callable, compress_fn: Callable,
                 bits_fn: Callable, device: str = "cpu"):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model_fn().to(device)
        self.compress_fn = compress_fn
        self.bits_fn = bits_fn
        self.device = device
        self.num_samples = len(local_data.dataset)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_weights(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def compute_gradient_on_batch(self, data: torch.Tensor, target: torch.Tensor
                                   ) -> Tuple[List[torch.Tensor], int]:
        """
        Compute gradient on ONE batch. No local model update.
        Mirrors TF GradientTape exactly.
        """
        self.model.train()
        self.model.zero_grad()

        data = data.to(self.device)
        target = target.to(self.device)

        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()

        gradients = []
        total_bits = 0

        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad.detach().clone()
                compressed = self.compress_fn(grad)
                gradients.append(compressed)
                total_bits += self.bits_fn(grad)
            else:
                gradients.append(torch.zeros_like(param))

        return gradients, total_bits

    def compute_fedavg_update(self, global_weights: Dict[str, torch.Tensor],
                               local_epochs: int = 1, local_lr: float = 0.01,
                               ) -> Tuple[Dict[str, torch.Tensor], int]:
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=local_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for data, target in self.local_data:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        local_weights = self.model.state_dict()
        gradient_update = {}
        total_bits = 0

        for key in global_weights:
            delta = global_weights[key].float() - local_weights[key].float()
            compressed_delta = self.compress_fn(delta)
            gradient_update[key] = compressed_delta
            total_bits += self.bits_fn(delta)

        return gradient_update, total_bits


# =============================================================================
# Experimental Unit
# =============================================================================

class ExperimentalUnit:
    """
    Orchestrates FL experiments.

    "raw" mode per epoch (TF behavior):
      1. Distribute global weights to all clients
      2. For each batch (synchronized):
         a. Each client computes gradient on its batch
         b. Server aggregates + applies (Adam)
         c. Server distributes updated weights to clients
      3. Evaluate

    "fedavg" mode per round:
      1. Distribute global weights
      2. Clients train locally for E epochs
      3. Server aggregates pseudo-gradients
      4. Evaluate
    """

    def __init__(self, server: FLServer, clients: List[FLClient],
                 test_loader: DataLoader):
        self.server = server
        self.clients = clients
        self.test_loader = test_loader

    def run(self, num_rounds: int, gradient_strategy: str = "raw",
            local_epochs: int = 1, local_lr: float = 0.01,
            eval_every: int = 1, verbose: bool = True) -> Dict:
        if gradient_strategy == "raw":
            return self._run_raw(num_rounds, eval_every, verbose)
        elif gradient_strategy == "fedavg":
            return self._run_fedavg(num_rounds, local_epochs, local_lr,
                                     eval_every, verbose)
        else:
            raise ValueError(f"Unknown strategy: '{gradient_strategy}'")

    def _run_raw(self, num_epochs: int, eval_every: int, verbose: bool) -> Dict:
        history = self._init_history()
        total_bits = 0

        # Initial eval
        init_loss, init_acc = self.server.evaluate(self.test_loader)
        if verbose:
            print(f"[Epoch 0] Loss: {init_loss:.4f} | Accuracy: {init_acc:.4f}")
        self._record(history, 0, init_loss, init_acc, 0, 0)

        for epoch in range(1, num_epochs + 1):
            epoch_bits = 0

            # 1. Distribute global weights to all clients
            global_weights = self.server.get_global_weights()
            for client in self.clients:
                client.set_weights(global_weights)

            # 2. Create synchronized batch iterators
            client_iterators = [iter(c.local_data) for c in self.clients]
            num_batches = min(len(c.local_data) for c in self.clients)

            # 3. Process each batch
            for batch_idx in range(num_batches):
                batch_gradients = []

                for c_idx, client in enumerate(self.clients):
                    try:
                        data, target = next(client_iterators[c_idx])
                    except StopIteration:
                        client_iterators[c_idx] = iter(client.local_data)
                        data, target = next(client_iterators[c_idx])

                    grads, bits = client.compute_gradient_on_batch(data, target)
                    batch_gradients.append(grads)
                    epoch_bits += bits

                # Server aggregates and applies (Adam)
                self.server.aggregate_and_apply(batch_gradients)

                # Distribute updated weights for next batch
                global_weights = self.server.get_global_weights()
                for client in self.clients:
                    client.set_weights(global_weights)

            total_bits += epoch_bits

            # Evaluate
            if epoch % eval_every == 0 or epoch == num_epochs:
                test_loss, test_acc = self.server.evaluate(self.test_loader)
                self._record(history, epoch, test_loss, test_acc,
                             epoch_bits, total_bits)
                if verbose:
                    bits_mb = total_bits / (8 * 1024 * 1024)
                    print(
                        f"[Epoch {epoch:3d}] "
                        f"Loss: {test_loss:.4f} | "
                        f"Acc: {test_acc:.4f} | "
                        f"Batches: {num_batches} | "
                        f"Total: {bits_mb:.2f} MB"
                    )

        return history

    def _run_fedavg(self, num_rounds: int, local_epochs: int,
                     local_lr: float, eval_every: int, verbose: bool) -> Dict:
        history = self._init_history()
        total_bits = 0

        init_loss, init_acc = self.server.evaluate(self.test_loader)
        if verbose:
            print(f"[Round 0] Loss: {init_loss:.4f} | Accuracy: {init_acc:.4f}")
        self._record(history, 0, init_loss, init_acc, 0, 0)

        for round_num in range(1, num_rounds + 1):
            global_weights = self.server.get_global_weights()
            client_updates = []
            client_weights = []
            round_bits = 0

            for client in self.clients:
                update, bits = client.compute_fedavg_update(
                    global_weights, local_epochs=local_epochs, local_lr=local_lr)
                client_updates.append(update)
                client_weights.append(client.num_samples)
                round_bits += bits

            total_bits += round_bits
            self.server.aggregate_dicts_and_apply(client_updates, client_weights)

            if round_num % eval_every == 0 or round_num == num_rounds:
                test_loss, test_acc = self.server.evaluate(self.test_loader)
                self._record(history, round_num, test_loss, test_acc,
                             round_bits, total_bits)
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

    @staticmethod
    def _init_history():
        return {
            "accuracy": [], "loss": [],
            "bits_per_round": [], "cumulative_bits": [],
            "rounds": [],
        }

    @staticmethod
    def _record(history, round_num, loss, acc, round_bits, cum_bits):
        history["rounds"].append(round_num)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        history["bits_per_round"].append(round_bits)
        history["cumulative_bits"].append(cum_bits)


# =============================================================================
# Convenience function
# =============================================================================

def fl_train(
    num_clients: int = 10,
    num_rounds: int = 50,
    local_epochs: int = 1,
    local_lr: float = 0.01,
    global_lr: float = 0.001,
    batch_size: int = 32,
    gradient_strategy: str = "raw",
    dataset_name: str = "mnist",
    partition_method: str = "iid",
    dirichlet_alpha: float = 0.5,
    max_samples: int = None,
    compression_method: str = "none",
    compression_kwargs: dict = None,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
    eval_every: int = 1,
) -> Dict:
    torch.manual_seed(seed)

    if compression_kwargs is None:
        compression_kwargs = {}

    from data import load_mnist, partition_data, get_client_loader, get_test_loader
    from models import get_model

    if dataset_name == "mnist":
        train_dataset, test_dataset = load_mnist()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_samples is not None and max_samples < len(train_dataset):
        from torch.utils.data import Subset
        indices = torch.randperm(len(train_dataset))[:max_samples].tolist()
        train_dataset = Subset(train_dataset, indices)

    client_datasets = partition_data(
        train_dataset, num_clients,
        method=partition_method, alpha=dirichlet_alpha, seed=seed
    )
    test_loader = get_test_loader(test_dataset)

    compress_fn, bits_fn = get_compressor(compression_method, **compression_kwargs)
    model_fn = lambda: get_model(dataset_name)

    if gradient_strategy == "raw":
        server = FLServer(model=model_fn(), lr=global_lr,
                          optimizer_type="adam", device=device)
    else:
        server = FLServer(model=model_fn(), lr=global_lr,
                          optimizer_type="sgd", device=device)

    clients = []
    for i in range(num_clients):
        client_loader = get_client_loader(client_datasets[i], batch_size)
        client = FLClient(
            client_id=i, local_data=client_loader, model_fn=model_fn,
            compress_fn=compress_fn, bits_fn=bits_fn, device=device,
        )
        clients.append(client)

    experiment = ExperimentalUnit(server, clients, test_loader)
    history = experiment.run(
        num_rounds=num_rounds, gradient_strategy=gradient_strategy,
        local_epochs=local_epochs, local_lr=local_lr,
        eval_every=eval_every, verbose=verbose,
    )

    history["config"] = {
        "num_clients": num_clients, "num_rounds": num_rounds,
        "local_epochs": local_epochs, "local_lr": local_lr,
        "global_lr": global_lr, "batch_size": batch_size,
        "gradient_strategy": gradient_strategy,
        "compression": compression_method,
        "compression_kwargs": compression_kwargs,
        "partition": partition_method, "dirichlet_alpha": dirichlet_alpha,
    }

    return history
