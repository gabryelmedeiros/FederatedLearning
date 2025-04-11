from train import simple_federated_emulation_per_batch as flb, centralized_learning as cl
import pandas as pd
import tensorflow as tf


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

  train_datasets, test_dataset = flb.get_and_divide_dataset(num_clients, batch_size, size)
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  central_unit = flb.CentralServer.__new__(flb.CentralServer)
  central_unit.__init__(cl.cnn, optimizer, loss_fn, ["accuracy"])
  central_unit.compile()

  clients = [flb.Client(cl.cnn, train_datasets[i], f"user00{i + 1}", optimizer, loss_fn, ["accuracy"]) for i in
             range(num_clients)]

  experimental_unit = flb.ExperimentalUnit(num_clients, test_dataset)
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
path = "./results/batch_size_comparison"

for size in BATCH_SIZE:
  print(f"Initializing federated emulation with size: {size}\n")
  df_batch_wise = federated_batch_training("../outputs/federated_emulation_per_batch", NUM_IMAGES, NUM_CLIENTS, NUM_ROUNDS, size)
  print(f"Saving results to: {path + f"batch_wise_size_{size}.csv"}\n")
  df_batch_wise.to_csv(path + f"batch_wise_size_{size}.csv")
  print("Results saved.\n")