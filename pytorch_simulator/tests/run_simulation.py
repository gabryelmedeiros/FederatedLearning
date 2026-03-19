import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pandas as pd
from datetime import datetime

from pytorch_simulator.simulator import fl_train

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Shared hyperparameters ────────────────────────────────────────────────────
DATASET      = "mnist"   # "mnist" or "cifar10"
NUM_CLIENTS  = 10
BATCH_SIZE   = 32
LOCAL_EPOCHS = 5
LOCAL_LR     = 0.01
SEED         = 42

# ── Round counts ──────────────────────────────────────────────────────────────
# Both modes run 20 rounds. Use cumulative_samples or cumulative_bits as the
# x-axis when comparing plots — not round number — since each round sees very
# different amounts of data (raw: ~320 samples, fedavg: ~60,000 samples).
RAW_ROUNDS    = 10
FEDAVG_ROUNDS = 10

# ── Output folder ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Experiments ───────────────────────────────────────────────────────────────
experiments = [
    {
        "name": "raw_no_compression",
        "gradient_strategy": "raw",
        "num_rounds": RAW_ROUNDS,
        "compression_method": "none",
        "lr": 0.01,
        "batch_size": 128
    },
    {
        "name": "fedavg_no_compression",
        "gradient_strategy": "fedavg",
        "num_rounds": FEDAVG_ROUNDS,
        "compression_method": "none",
        "lr": LOCAL_LR,
        "batch_size": BATCH_SIZE
    },
]

# ── Run ───────────────────────────────────────────────────────────────────────
for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}  [{DATASET}]  ({exp['num_rounds']} rounds)")
    print(f"{'='*60}")

    history = fl_train(
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        num_rounds=exp["num_rounds"],
        batch_size=BATCH_SIZE,
        local_epochs=LOCAL_EPOCHS,
        local_lr=exp["lr"],
        gradient_strategy=exp["gradient_strategy"],
        compression_method=exp["compression_method"],
        compression_kwargs=exp.get("compression_kwargs", {}),
        device=device,
        seed=SEED,
        verbose=True,
        eval_every=1,
    )

    df = pd.DataFrame({
        "round":              history["rounds"],
        "accuracy":           history["accuracy"],
        "loss":               history["loss"],
        "bits_per_round":     history["bits_per_round"],
        "cumulative_bits":    history["cumulative_bits"],
        "samples_per_round":  history["samples_per_round"],
        "cumulative_samples": history["cumulative_samples"],
    })

    csv_path = os.path.join(RESULTS_DIR, f"{DATASET}_{exp['name']}_{RUN_TS}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Final accuracy:     {history['accuracy'][-1]:.4f}")
    print(f"Total bits:         {history['cumulative_bits'][-1] / 8e6:.2f} MB")
    print(f"Total samples seen: {history['cumulative_samples'][-1]:,}")