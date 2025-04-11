## importing libraries
from train import centralized_learning as cl, simple_federated_emulation_per_batch as flb, \
  simple_federated_emulation_per_average as fl
import pandas as pd
import tensorflow as tf


## creating funcions to run each type of training
def centralized_training(dataset_size: int, buffer_size: int, batch_size: int, num_epochs: int) -> pd.DataFrame:
  """
    Uses the library centralized_learning to execute a train session using the given parameters.

    Args:
        dataset_size (int): Number of images to create the dataset
        buffer_size (int): Size of the buffer
        batch_size (int): Size of each batch
        num_epochs (int): Number of epochs to train.

    Returns:
        pd.DataFrame: A DataFrame containing training metrics.
    """

  model = cl.cnn()
  x_train, y_train, x_test, y_test = cl.load_and_process_dataset(dataset_size)
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

  metrics_df = cl.train_model(model, train_dataset, epochs=num_epochs)

  return metrics_df


def federated_average_training(path: str, size: int, num_clients: int, rounds: int, batch_size: int) -> pd.DataFrame:
  """
    Uses the library simple_federated_emulation to execute a train session using the given parameters.

    Args:
        path (str): Path where the gradients will be stored
        size (int): Number of images to create the dataset
        num_clients (int): Number of clients for the emulation
        batch_size (int): Size of each batch
        rounds (int): Number of rounds to train.

    Returns:
        pd.DataFrame: A DataFrame containing training metrics.
  """

  train_datasets, test_dataset = fl.get_and_divide_dataset(num_clients, batch_size, size)
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  central_unit = fl.CentralServer(cl.cnn, optimizer, loss_fn, ["accuracy"])
  central_unit.compile()

  clients = [
      fl.Client(cl.cnn, train_datasets[j], f"user00{j+1}", optimizer, loss_fn, ["accuracy"]) for j in range(num_clients)
  ]

  experimental_unit = fl.ExperimentalUnit(num_clients)
  metrics_df = experimental_unit.federated_training(central_unit, clients, test_dataset, path, rounds)

  return metrics_df

def federated_batch_training(path: str, size: int, num_clients:int, rounds:int, batch_size: int) -> pd.DataFrame:
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

  clients =[flb.Client(cl.cnn, train_datasets[i], f"user00{i+1}", optimizer, loss_fn, ["accuracy"]) for i in range(num_clients)]

  experimental_unit = flb.ExperimentalUnit(num_clients, test_dataset)
  metrics_df = experimental_unit.federated_training(
    central_unit=central_unit,
    clients=clients,
    rounds=rounds,
    batch_size=batch_size,
    gradients_path=path,
  )

  return metrics_df

# defining parameters
NUM_IMAGES = 2000
NUM_CLIENTS = 10
NUM_EPOCHS = 30
BATCH_SIZE = 32
BUFFER_SIZE = 1024
EPOCHS = 10
MINI_BATCH_SIZE = 32

print(f"Number of images: {NUM_IMAGES}")
print(f"Number of clients: {NUM_CLIENTS}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Mini-batch size: {MINI_BATCH_SIZE}")

# starting trainings
print("\n\nStarting trainings:\n")
df_batch_wise = federated_batch_training("../outputs/federated_emulation_per_batch", NUM_IMAGES, NUM_CLIENTS, NUM_EPOCHS, MINI_BATCH_SIZE)
df_average = federated_average_training("../outputs/federated_emulation_average", NUM_IMAGES, NUM_CLIENTS, NUM_EPOCHS, BATCH_SIZE)
df_centralized = centralized_training(NUM_IMAGES, BUFFER_SIZE, BATCH_SIZE, EPOCHS)

path = "./results/"
df_batch_wise.to_csv(path + "batch_wise.csv")
df_average.to_csv(path + "average.csv")
df_centralized.to_csv(path + "centralized.csv")