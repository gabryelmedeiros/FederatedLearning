"""
Weight Compression Experiment — MPGBP applied to trained model weights.

Pipeline:
  1. Split MNIST into train (50k) / val (10k) / test (10k)
  2. Train MNISTNet; save best checkpoint by validation accuracy
  3. Evaluate best model on test set (baseline, no compression)
  4. For each (mode, M_max):
       - deepcopy the original trained weights
       - apply MPGBP compression
       - evaluate on test set
  5. Save results to CSV

Modes:
  V (per-Vector)  : each layer tensor flattened into one vector
  C (per-Channel) : each row/channel of each tensor independently
  R (per-network) : ALL parameters concatenated into one vector  ← slow
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd

from pytorch_simulator.models import get_model, count_parameters
from pytorch_simulator.data import load_mnist, get_test_loader
from pytorch_simulator.compressors import (mpgbp, compress_mpgbp, apply_mpgbp,
                                           spt_count, extract_exponents)


# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EPOCHS = 10
BATCH_SIZE   = 64
LR           = 0.001
VAL_SPLIT    = 10_000
# M_max scaling: pass a multiplier; actual M_max computed per-tensor inside compress_mpgbp.
# S mode always uses M_max=1 (fixed). R, L, N use multiplier-based scaling.
MULTIPLIERS_R = [0.5, 1, 1.5, 2]
MULTIPLIERS_L = [0.5, 0.75, 1, 1.25]
MULTIPLIERS_N = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
# MODES         = ["S", "R", "L", "N"]
MODES = ["N"]
SEED         = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
print(f"Device: {DEVICE}\n")


# ── Data split ─────────────────────────────────────────────────────────────────
train_full, test_dataset = load_mnist()

train_dataset, val_dataset = random_split(
    train_full,
    [len(train_full) - VAL_SPLIT, VAL_SPLIT],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=256,        shuffle=False)
test_loader  = get_test_loader(test_dataset)

print(f"Train samples : {len(train_dataset):,}")
print(f"Val samples   : {len(val_dataset):,}")
print(f"Test samples  : {len(test_dataset):,}\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.to(device).eval()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out         = model(data)
            total_loss += criterion(out, target).item() * len(data)
            correct    += out.argmax(1).eq(target).sum().item()
            total      += len(data)
    return total_loss / total, correct / total


# ── Compression ────────────────────────────────────────────────────────────────
def _mmax_info(model, multiplier, mode):
    """Return a string describing the M_max values that will be used."""
    lines = []
    for name, param in model.named_parameters():
        t = param.data
        if mode == "S":
            lines.append(f"  {name}: M_max=1 (fixed)")
        elif mode == "R":
            K = t[0].numel() if t.dim() >= 2 else t.numel()
            m = max(1, math.ceil(multiplier * K / 2))
            lines.append(f"  {name}: K={K}, M_max={m}")
        elif mode == "L":
            if t.dim() == 1:
                K, R = t.numel(), 1
            else:
                K = t[0].numel()
                R = t.shape[0]
            m = max(1, math.ceil(multiplier * K * R / 4))
            lines.append(f"  {name}: K={K}, R={R}, M_max={m}")
        elif mode == "N":
            pass  # single M_max for whole network, handled separately
    return "\n".join(lines)


def compute_mmax_range(model, multiplier, mode):
    """Return (m_max_min, m_max_max) for the given mode and multiplier."""
    if mode == "S" or multiplier is None:
        return 1, 1
    elif mode == "R":
        values = []
        for param in model.parameters():
            t = param.data
            K = t[0].numel() if t.dim() >= 2 else t.numel()
            values.append(max(1, math.ceil(multiplier * K / 2)))
        return min(values), max(values)
    elif mode == "L":
        values = []
        for param in model.parameters():
            t = param.data
            if t.dim() == 1:
                K, R = t.numel(), 1
            else:
                K = t[0].numel()
                R = t.shape[0]
            values.append(max(1, math.ceil(multiplier * K * R / 4)))
        return min(values), max(values)
    elif mode == "N":
        param_list = list(model.parameters())
        N = sum(p.numel() for p in param_list)
        L = len(param_list)
        m = max(1, math.ceil(multiplier * N / (2 * L)))
        return m, m
    return None, None


def compress_model(original_model, multiplier, mode):
    """
    Return (new_model, metadata) where new_model has MPGBP-compressed weights.
    original_model is never mutated.
    Metadata aggregates total_spts, max_exponent, num_iterations, residual_norm
    across all parameter tensors.
    """
    model = copy.deepcopy(original_model)
    total_meta = {"total_spts": 0, "max_exponent": 0,
                  "num_iterations": 0, "residual_norm": 0.0}

    def _merge(dst, src):
        dst["total_spts"]     += src["total_spts"]
        dst["max_exponent"]    = max(dst["max_exponent"], src["max_exponent"])
        dst["num_iterations"] += src["num_iterations"]
        dst["residual_norm"]  += src["residual_norm"]

    if mode == "S":
        with torch.no_grad():
            for param in model.parameters():
                compressed, meta = compress_mpgbp(
                    param.data.cpu(), M_max=1, mode="S", return_metadata=True
                )
                param.data = compressed.to(param.dtype).to(param.device)
                _merge(total_meta, meta)

    elif mode in ("R", "L"):
        with torch.no_grad():
            for param in model.parameters():
                compressed, meta = compress_mpgbp(
                    param.data.cpu(), multiplier=multiplier, mode=mode,
                    return_metadata=True
                )
                param.data = compressed.to(param.dtype).to(param.device)
                _merge(total_meta, meta)

    elif mode == "N":
        param_list = list(model.parameters())
        N = sum(p.numel() for p in param_list)
        L = len(param_list)
        m = max(1, math.ceil(multiplier * N / (2 * L)))
        print(f"    N mode: total_params={N:,}, L={L}, M_max={m}")
        params_in = [(p.data.cpu(), p.shape) for p in param_list]
        compressed = apply_mpgbp(params_in, multiplier=multiplier)
        with torch.no_grad():
            for param, comp in zip(param_list, compressed):
                param.data = comp.to(param.dtype).to(param.device)
        # N mode: estimate metadata from compressed model parameters
        all_params_flat = torch.cat([p.data.cpu().reshape(-1).float()
                                     for p in model.parameters()])
        total_meta["total_spts"] = int(spt_count(all_params_flat).item())
        all_exp = extract_exponents(all_params_flat)
        total_meta["max_exponent"] = max(abs(e) for e in all_exp) if all_exp else 0

    return model, total_meta


# ── 1. Train ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Training baseline model…")
print("=" * 60)

model     = get_model("mnist").to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
best_state   = None

for epoch in range(1, TRAIN_EPOCHS + 1):
    model.train()
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

    _, val_acc = evaluate(model, val_loader, DEVICE)
    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = copy.deepcopy(model.state_dict())
        marker       = "  ← best"
    print(f"  [Epoch {epoch:2d}/{TRAIN_EPOCHS}]  Val acc: {val_acc:.4f}{marker}")

model.load_state_dict(best_state)
print(f"\nBest val accuracy : {best_val_acc:.4f}")

# ── 2. Baseline ────────────────────────────────────────────────────────────────
_, baseline_acc = evaluate(model, test_loader, DEVICE)

print(f"\n{'─'*60}")
print(f"Baseline test accuracy : {baseline_acc:.4f}")
print(f"Parameters             : {count_parameters(model):,}")
print(f"{'─'*60}\n")

# ── 3. Compression sweep ───────────────────────────────────────────────────────
mode_multipliers = {
    "S": [None],           # always M_max=1
    "R": MULTIPLIERS_R,
    "L": MULTIPLIERS_L,
    "N": MULTIPLIERS_N,
}

rows = []

for mode in MODES:
    for multiplier in mode_multipliers[mode]:
        label = "M_max=1 (fixed)" if mode == "S" else f"multiplier={multiplier}"
        print(f"[Mode={mode}, {label}]")

        if mode not in ("S", "N"):
            print(_mmax_info(model, multiplier, mode))

        m_max_min, m_max_max = compute_mmax_range(model, multiplier, mode)

        t0 = time.time()
        compressed_model, meta = compress_model(model, multiplier, mode)
        elapsed                = time.time() - t0
        _, acc                 = evaluate(compressed_model, test_loader, DEVICE)

        print(f"  Test acc     : {acc:.4f}  (drop: {baseline_acc - acc:+.4f})")
        print(f"  Total SPTs   : {meta['total_spts']:,}")
        print(f"  Max exponent : {meta['max_exponent']}")
        print(f"  Iterations   : {meta['num_iterations']:,}")
        print(f"  Time         : {elapsed:.1f}s\n")

        rows.append({
            "mode":           mode,
            "multiplier":     multiplier if multiplier is not None else "fixed",
            "m_max_min":      m_max_min,
            "m_max_max":      m_max_max,
            "test_accuracy":  round(acc, 6),
            "accuracy_drop":  round(baseline_acc - acc, 6),
            "total_spts":     meta["total_spts"],
            "max_exponent":   meta["max_exponent"],
            "num_iterations": meta["num_iterations"],
            "time_s":         round(elapsed, 2),
        })

# ── 4. Save & print ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, f"weight_compression_net_only_{RUN_TS}.csv") #change here as well after testing only the new.
df.to_csv(csv_path, index=False)

print("=" * 60)
print(f"Baseline test accuracy: {baseline_acc:.4f}")
print("=" * 60)
print(df.to_string(index=False))
print(f"\nSaved → {csv_path}")