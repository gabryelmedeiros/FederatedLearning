import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train.simple_federated_emulation_per_batch
import train.centralized_learning
import pandas as pd
import tensorflow as tf

from experiments.comparing_training_methods import centralized_training


def federated_batch_training(path: str, size: int, num_clients: int, rounds: int, batch_size: int) -> pd.DataFrame:
  """
  Uses the library simple_federated_emulation_per_batch to execute a train session using the given parameters.

  Args:
      path (str): Path where the gradients will be stored
      size (int): Number of images to create the dataset
      num_clients (int): Number of clients for the emulation
      batch_size (int): Size of each batch
      rounds (int): Number of rounds to train.

  Returns:
      pd.DataFrame: A DataFrame containing training metrics.
"""

  train_datasets, test_dataset = train.simple_federated_emulation_per_batch.get_and_divide_dataset(num_clients, batch_size, size)
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  central_unit = train.simple_federated_emulation_per_batch.CentralServer.__new__(train.simple_federated_emulation_per_batch.CentralServer)
  central_unit.__init__(train.centralized_learning.cnn, optimizer, loss_fn, ["accuracy"])
  central_unit.compile()

  clients = [train.simple_federated_emulation_per_batch.Client(train.centralized_learning.cnn, train_datasets[i], f"user00{i + 1}", optimizer, loss_fn, ["accuracy"]) for i in
             range(num_clients)]

  experimental_unit = train.simple_federated_emulation_per_batch.ExperimentalUnit(num_clients, test_dataset)
  metrics_df = experimental_unit.federated_training(
    central_unit,
    clients,
    rounds,
    batch_size,
    path
  )

  return metrics_df

# defining parameters
NUM_IMAGES = 2000
NUM_CLIENTS = 10
NUM_ROUNDS = 30
BATCH_SIZE = [3, 6, 9 ,12, 15, 18, 21, 24, 27, 30]
path = "./results/batch_size_comparison_"

for size in BATCH_SIZE:
  print(f"Initializing federated emulation with size: {size}\n")
  df_batch_wise = federated_batch_training("../outputs/federated_emulation_per_batch", NUM_IMAGES, NUM_CLIENTS, NUM_ROUNDS, size)
  print(f"Saving results to: {path + f"size_{size}.csv"}\n")
  df_batch_wise.to_csv(path + f"size_{size}.csv")
  print("Results saved.\n")