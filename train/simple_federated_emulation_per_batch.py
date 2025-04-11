import tensorflow as tf
import numpy as np
import os
import pandas as pd
from train.centralized_learning import cnn


# Central Server Class

class CentralServer:
    """
    Central server in a federated learning system with batch-wise gradient aggregation.

    This class manages the global model, aggregates gradients received from clients
    (one batch per client per round), and evaluates the global model.
    """
    def __init__(self, model_function, optimizer, loss, metrics):
        """
        Initializes the CentralServer.

        Args:
            model_function: A function that returns a TensorFlow Keras model.
            optimizer: A TensorFlow Keras optimizer instance.
            loss: A TensorFlow Keras loss function.
            metrics: A list of TensorFlow Keras metrics.
        """
        self.global_model = model_function()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compile(self):
        """
        Compiles the global model with the specified optimizer, loss, and metrics.
        """
        self.global_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def send_model(self):
        """
        Returns the current weights of the global model.

        Returns:
            A list of NumPy arrays representing the model's weights.
        """
        return self.global_model.get_weights()

    def get_update_from_user(self, user, gradients_path, num_batches):
        """
        Retrieves gradient updates from a specific user (intended for full gradient transfer,
        might not be directly used in the current batch-wise `ExperimentalUnit`).

        Assumes that the user has saved their gradients per batch as binary files
        in a directory named with their user ID under the given `gradients_path`.

        Args:
            user: An object representing the user, which should have a `get_id()` method
                  and a `get_model_weights()` method returning the shapes of the weights.
            gradients_path: The base path where user gradient directories are stored.
            num_batches: The number of batches in the user's training dataset.

        Returns:
            A list of lists of NumPy arrays representing the user's gradients for each batch,
            or None if there's an error reading the gradient files.
        """
        all_gradients = []
        user_id = user.get_id()
        user_model_shapes = [w.shape for w in user.get_model_weights()]

        @staticmethod
        def _read_binary_file(file_path, shape):
            try:
                with open(file_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.float32)
                return data.reshape(shape)
            except FileNotFoundError:
                print(f"Error: Gradient file not found at {file_path}")
                return None
            except IOError as e:
                print(f"Error reading file {file_path}: {e}")
                return None

        for step in range(num_batches):
            batch_gradients = []
            user_batch_path = os.path.join(gradients_path, str(user_id))
            for i, shape in enumerate(user_model_shapes):
                layer_type = ""
                if len(shape) == 1:
                    layer_type = "bias"
                elif len(shape) == 2:
                    layer_type = "dense"
                elif len(shape) == 4:
                    layer_type = "conv"
                else:
                    layer_type = f"layer_{i+1}"

                file_name = f'batch_{step}_/{layer_type}{i+1}.bin' # Assuming subdirectories for batches
                file_path = os.path.join(user_batch_path, file_name)
                gradient_data = _read_binary_file(file_path, shape)
                if gradient_data is not None:
                    batch_gradients.append(gradient_data)
                else:
                    print(f"Warning: Could not read all gradients for user {user_id}, batch {step}.")
                    return None # Or consider a different error handling strategy
            if batch_gradients:
                all_gradients.append(batch_gradients)
            else:
                return None # Or consider a different error handling strategy

        return all_gradients

    def aggregate(self, client_gradients):
        """
        Aggregates the gradients received from clients (one batch per client).

        Args:
            client_gradients: A list of lists, where each inner list contains
                              the gradients from one client for their current batch.
                              Each client's gradients should have the same structure
                              (same number and shapes of arrays) as the global model's weights.

        Returns:
            A list of NumPy arrays representing the averaged gradients.
            Returns a list of zeros if no client gradients are provided.
        """
        if not client_gradients:
            print("Warning: No client gradients provided for aggregation. Returning zeros.")
            return [tf.zeros_like(w).numpy() for w in self.global_model.trainable_weights]

        # Initialize with the first client's gradients as NumPy arrays
        aggregated_gradients = [np.array(grad) for grad in client_gradients[0]]

        # Accumulate gradients from other clients
        for i in range(1, len(client_gradients)):
            for j in range(len(aggregated_gradients)):
                aggregated_gradients[j] += np.array(client_gradients[i][j])

        # Calculate the mean
        num_clients = len(client_gradients)
        averaged_gradients = [grad / num_clients for grad in aggregated_gradients]

        return averaged_gradients

    @tf.function
    def apply_gradients(self, grads):
        """
        Applies the aggregated gradients to the global model.

        Args:
            grads: A list of NumPy arrays representing the aggregated gradients.
        """
        # Convert NumPy arrays to TensorFlow Tensors
        tensor_grads = [tf.convert_to_tensor(g) for g in grads]
        self.optimizer.apply_gradients(zip(tensor_grads, self.global_model.trainable_weights))

    def evaluate(self, test_dataset):
        """
        Evaluates the global model on the given test dataset.

        Args:
            test_dataset: A TensorFlow Dataset object for testing.

        Returns:
            A tuple containing the average loss and the accuracy on the test dataset.
        """
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        total_loss = tf.keras.metrics.Mean(name='test_loss')

        @tf.function
        def evaluation_step(test_images, test_labels):
            predictions = self.global_model(test_images, training=False)
            loss = self.loss(test_labels, predictions)
            total_loss.update_state(loss)
            accuracy_metric.update_state(test_labels, predictions)

        for test_images, test_labels in test_dataset:
            evaluation_step(test_images, test_labels)

        accuracy_result = accuracy_metric.result()
        average_loss = total_loss.result()
        return average_loss, accuracy_result


# Client Class
class Client:
    """
    Represents a client in a federated learning system with batch-wise gradient transmission.

    Each client has a local model, a training dataset, and methods to compute
    and save gradients per batch.
    """
    def __init__(self, function, train_dataset, id, optimizer, loss, metrics):
        """
        Initializes a Client instance.

        Args:
            function: A function that returns a TensorFlow Keras model.
            train_dataset: A TensorFlow Dataset object for training.
            id: The unique identifier of the client.
            optimizer: A TensorFlow Keras optimizer instance.
            loss: A TensorFlow Keras loss function.
            metrics: A list of TensorFlow Keras metrics.
        """
        self.local_model = function()
        self.train_dataset = train_dataset
        self.id = id
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def get_id(self):
        """Returns the client's ID."""
        return self.id

    def compile(self):
        """Compiles the local model with the specified optimizer, loss, and metrics."""
        self.local_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def set_model_weights(self, weights):
        """Sets the weights of the local model."""
        self.local_model.set_weights(weights)

    def get_model_weights(self):
        """Returns the current weights of the local model."""
        return self.local_model.get_weights()

    @tf.function
    def get_gradients(self, images, labels):
        """Computes the gradients of the local model's loss with respect to its weights for a batch."""
        with tf.GradientTape() as tape:
            logits = self.local_model(images, training=True)
            loss_value = self.loss_fn(labels, logits)
        gradients = tape.gradient(loss_value, self.local_model.trainable_weights)
        return gradients

    def _save_array(self, file_path, array):
        """Utility function to save a NumPy array to a binary file."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            with open(file_path, "wb") as f:
                array.astype(np.float32).tofile(f)
        except IOError as e:
            print(f"Error saving file {file_path}: {e}")

    def save_local_gradients_per_batch(self, path):
        """Saves the gradients for each batch of the local training dataset to separate binary files."""
        for step, (x_batch, y_batch) in enumerate(self.train_dataset):
            gradients = self.get_gradients(x_batch, y_batch)
            for i, gradient in enumerate(gradients):
                gradient_np = gradient.numpy()
                file_name = ""
                if len(gradient_np.shape) == 1:
                    file_name = f'batch_{step}_bias{i+1}.bin'
                elif len(gradient_np.shape) == 2:
                    file_name = f'batch_{step}_dense{i+1}.bin'
                elif len(gradient_np.shape) == 4:
                    file_name = f'batch_{step}_conv{i+1}.bin'
                else:
                    file_name = f'batch_{step}_layer_{i+1}.bin'

                file_path = os.path.join(path, self.id, file_name)
                self._save_array(file_path, gradient_np)


# Experimental Unit
class ExperimentalUnit:
    """
    Orchestrates the federated learning experiment with batch-wise gradient transmission.

    In each round, each client processes one mini-batch, and the gradients are aggregated.
    """
    def __init__(self, num_clients, test_dataset):
        """
        Initializes the ExperimentalUnit.

        Args:
            num_clients: The number of clients in the experiment.
            test_dataset: The test dataset for evaluation of the global model.
        """
        self.num_clients = num_clients
        self.test_dataset = test_dataset

    def federated_training(self, central_unit, clients, rounds, batch_size, gradients_path):
        """
        Performs federated training for a specified number of epochs, where each
        client contributes gradients from one mini-batch per round.

        Args:
            central_unit: The CentralServer instance.
            clients: A list of Client instances.
            rounds: The number of training epochs (rounds).
            batch_size: The batch size used by the clients.
            gradients_path: The path where clients save their gradients (not directly used here in the aggregation).

        Returns:
            A Pandas DataFrame containing the training metrics (total samples processed, accuracy, loss).
        """
        metrics = {}
        total_samples_processed = 0
        accuracy_list = []
        loss_list = []

        print("Iniciando treinamento federado (batch-wise)...\n")

        # Create iterators for the training datasets of each client
        train_iterators = [iter(client.train_dataset) for client in clients]

        for round in range(rounds):
            print(f"Rodada {round + 1} iniciada.")

            # Distribute global weights to all clients
            global_weights = central_unit.send_model()
            for client in clients:
                client.set_model_weights(global_weights)

            # Each client processes one mini-batch and sends gradients
            batch_gradients = []
            for client_index, client in enumerate(clients):
                try:
                    # Get the next mini-batch from the client
                    x_batch, y_batch = next(train_iterators[client_index])
                    print(f"Cliente {client.get_id()} processando um mini-batch.")
                    # Calculate gradients for the current mini-batch
                    gradients = client.get_gradients(x_batch, y_batch)
                    batch_gradients.append(gradients)
                except StopIteration:
                    # If the client's dataset is exhausted, reset the iterator
                    train_iterators[client_index] = iter(client.train_dataset)
                    x_batch, y_batch = next(train_iterators[client_index])
                    print(f"Cliente {client.get_id()} (dataset reset) processando um mini-batch.")
                    gradients = client.get_gradients(x_batch, y_batch)
                    batch_gradients.append(gradients)

            # Aggregate the gradients and update the global model
            print("Agregando gradientes...")
            average = central_unit.aggregate(batch_gradients)
            print("Aplicando gradientes ao modelo global...")
            central_unit.apply_gradients(average)

            # Update the total number of samples processed
            total_samples_processed += batch_size * len(clients)

            # Evaluate the global model after each round
            loss, accuracy = central_unit.evaluate(self.test_dataset)

            # Update metrics
            accuracy_list.append(accuracy.numpy())
            loss_list.append(loss.numpy())
            metrics[round] = {'loss': loss.numpy(), 'accuracy': accuracy.numpy()}

            # Display current round metrics
            print(f"Métricas da Rodada {round + 1}:")
            print(f"  Loss: {loss.numpy():.4f}")
            print(f"  Acurácia: {accuracy.numpy():.4f}\n")

        # Prepare data for plotting
        plot_data = {
            'total_samples_processed': [i * batch_size * len(clients) for i in range(1, rounds + 1)],
            'accuracy': accuracy_list,
            'loss': loss_list
        }

        df = pd.DataFrame(data=plot_data)

        return df

def get_and_divide_dataset(num_clients, batch_size, max_dataset_size, normalization_factor=255.0):
    """
    Loads the MNIST dataset, preprocesses it, and divides the training data among clients.

    Args:
        num_clients: The number of clients to divide the training data among.
        batch_size: The batch size for training and testing.
        max_dataset_size: The maximum size of the dataset to use.
        normalization_factor: The value to divide pixel values by for normalization.

    Returns:
        A tuple containing a list of TensorFlow Dataset objects for training (one per client)
        and a TensorFlow Dataset object for testing.
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    if num_clients <= 0:
        raise ValueError("Number of clients must be greater than zero.")

    if max_dataset_size > len(train_images):
        print(f"Warning: max_dataset_size {max_dataset_size} is bigger than the train dataset size. Setting it to train dataset size: {len(train_images)}")
        max_dataset_size = len(train_images)

    if max_dataset_size % num_clients != 0:
        raise ValueError("max_dataset_size must be divisible by num_clients.")

    train_images = train_images[:max_dataset_size]
    train_labels = train_labels[:max_dataset_size]
    test_images = test_images[:max_dataset_size]
    test_labels = test_labels[:max_dataset_size]

    train_images = train_images / normalization_factor
    test_images = test_images / normalization_factor

    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    images_per_client = max_dataset_size // num_clients
    train_datasets = []

    for i in range(num_clients):
        client_train_data = tf.data.Dataset.from_tensor_slices(
            (train_images[i * images_per_client:(i + 1) * images_per_client],
             train_labels[i * images_per_client:(i + 1) * images_per_client])
        )
        train_datasets.append(client_train_data.batch(batch_size))  # Removed drop_remainder=True

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    return train_datasets, test_dataset

# Example Usage
if __name__ == "__main__":
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    gradients_path = os.path.join(output_dir, "federated_emulation_per_batch")

    num_clients = 10
    batch_size = 32
    total_data_size = 2000
    epochs = 30

    train_datasets, test_dataset = get_and_divide_dataset(num_clients, batch_size, total_data_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    central_unit = CentralServer(cnn, optimizer, loss_function, metrics)
    central_unit.compile()

    clients = [Client(cnn, train_datasets[i], f"client_{i+1}", optimizer, loss_function, metrics) for i in range(num_clients)]

    experimental_unit = ExperimentalUnit(num_clients, test_dataset)
    metrics_df = experimental_unit.federated_training(central_unit, clients, epochs, batch_size, gradients_path)
    print(metrics_df)

    output_csv_path = os.path.join(output_dir, f"federated_batch_wise_size_{total_data_size}_clients_{num_clients}_epochs_{epochs}_batch_{batch_size}.csv")
    try:
        metrics_df.to_csv(output_csv_path, index=False)
        print(f"Metrics saved to {output_csv_path}")
    except IOError as e:
        print(f"Error saving metrics to CSV: {e}")