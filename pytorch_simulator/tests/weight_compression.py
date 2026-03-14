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
from pytorch_simulator.compressors import mpgbp, compress_mpgbp, apply_mpgbp


# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EPOCHS = 10
BATCH_SIZE   = 64
LR           = 0.001
VAL_SPLIT    = 10_000
M_MAX_VALUES = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
M_MAX_S      = [1, 2, 3, 4]   # S mode saturates quickly; no need for large values
MODES        = ["S", "R", "L", "N"]
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
def compress_model(original_model, M_max, mode):
    """Return a new model with MPGBP-compressed weights. Never mutates original."""
    model = copy.deepcopy(original_model)

    if mode in ("S", "R", "L"):
        with torch.no_grad():
            for param in model.parameters():
                param.data = compress_mpgbp(
                    param.data.cpu(), M_max=M_max, mode=mode
                ).to(param.dtype).to(param.device)

    elif mode == "N":
        param_list = list(model.parameters())
        N = sum(p.numel() for p in param_list)
        P = max(1, math.ceil(math.sqrt(N)))
        print(f"    N mode: N={N:,}, P={P:,}  (this may take a while…)")
        params_in = [(p.data.cpu(), p.shape) for p in param_list]
        compressed = apply_mpgbp(params_in, M_max=M_max)
        with torch.no_grad():
            for param, comp in zip(param_list, compressed):
                param.data = comp.to(param.dtype).to(param.device)

    return model


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
rows = []

for mode in MODES:
    m_max_list = M_MAX_S if mode == "S" else M_MAX_VALUES
    for M_max in m_max_list:
        print(f"[Mode={mode}, M_max={M_max}]")
        t0 = time.time()

        compressed_model = compress_model(model, M_max, mode)
        elapsed          = time.time() - t0
        _, acc           = evaluate(compressed_model, test_loader, DEVICE)

        print(f"  Test acc : {acc:.4f}  (drop: {baseline_acc - acc:+.4f})  [{elapsed:.1f}s]\n")

        rows.append({
            "mode":          mode,
            "M_max":         M_max,
            "test_accuracy": round(acc, 6),
            "accuracy_drop": round(baseline_acc - acc, 6),
            "time_s":        round(elapsed, 2),
        })

# ── 4. Save & print ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, f"weight_compression_{RUN_TS}.csv")
df.to_csv(csv_path, index=False)

print("=" * 60)
print(f"Baseline test accuracy: {baseline_acc:.4f}")
print("=" * 60)
print(df.to_string(index=False))
print(f"\nSaved → {csv_path}")