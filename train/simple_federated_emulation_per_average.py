import tensorflow as tf
import numpy as np
import os
import pandas as pd
from train.centralized_learning import cnn


# Central Server Class
class CentralServer:
    """
    Central server in a federated learning system.

    This class manages the global model, aggregates updates from users,
    and evaluates the global model.
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

    def set_weights(self, weights):
        """
        Sets the weights of the global model.

        Args:
            weights: A list of NumPy arrays representing the new model weights.
        """
        self.global_model.set_weights(weights)

    def _read_binary_gradient(self, file_path, shape):
        """
        Reads a binary file containing gradient data and reshapes it.

        Args:
            file_path: Path to the binary file.
            shape: The expected shape of the gradient data.

        Returns:
            A NumPy array representing the gradient, or None if the file is not found.
        """
        try:
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
            return data.reshape(shape)
        except FileNotFoundError:
            print(f"Error: Gradient file not found at {file_path}")
            return None

    def get_update_from_user(self, user, gradients_path):
        """
        Retrieves the gradient updates from a specific user.

        Assumes that the user has saved their gradients as binary files
        in a directory named with their user ID under the given `gradients_path`.
        The gradient files are expected to be named following a convention
        like 'bias{index}.bin', 'dense{index}.bin', or 'conv{index}.bin'.

        Args:
            user: An object representing the user, which should have methods
                  `get_id()` to get the user's ID and `get_model_weights()`
                  to get the shapes of the user's model weights.
            gradients_path: The base path where user gradient directories are stored.

        Returns:
            A list of NumPy arrays representing the user's gradients, or None
            if there's an error reading the gradient files.
        """
        user_id = user.get_id()
        user_gradient_dir = os.path.join(gradients_path, str(user_id))
        user_model_shapes = [w.shape for w in user.get_model_weights()]
        read_gradients = []

        for i, shape in enumerate(user_model_shapes):
            layer_type = ""
            if len(shape) == 1:
                layer_type = "bias"
            elif len(shape) == 2:
                layer_type = "dense"
            elif len(shape) == 4:
                layer_type = "conv"
            else:
                layer_type = f"layer_{i+1}" # Fallback name

            file_name = f'{layer_type}{i+1}.bin'
            file_path = os.path.join(user_gradient_dir, file_name)
            gradient_data = self._read_binary_gradient(file_path, shape)
            if gradient_data is not None:
                read_gradients.append(gradient_data)
            else:
                return None # Or handle the error appropriately

        return read_gradients

    def aggregate(self, user_gradients):
        """
        Aggregates the gradients received from multiple users by averaging them.

        Args:
            user_gradients: A list of lists, where each inner list contains
                            the gradients from one user. Each user's gradients
                            should have the same structure (same number and shapes
                            of arrays) as the global model's weights.

        Returns:
            A list of NumPy arrays representing the averaged gradients.
            Returns a list of zeros if no user gradients are provided.
        """
        if not user_gradients:
            print("Warning: No user gradients provided for aggregation. Returning zeros.")
            return [tf.zeros_like(w).numpy() for w in self.global_model.trainable_weights]

        # Initialize with the first user's gradients as NumPy arrays
        aggregated_gradients = [np.array(grad) for grad in user_gradients[0]]

        # Accumulate gradients from other users
        for i in range(1, len(user_gradients)):
            for j in range(len(aggregated_gradients)):
                aggregated_gradients[j] += np.array(user_gradients[i][j])

        # Calculate the mean
        num_users = len(user_gradients)
        averaged_gradients = [grad / num_users for grad in aggregated_gradients]

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
    Represents a client in a federated learning system.

    Each client has a local model, a training dataset, and methods to compute
    and save gradients.
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
        """Computes the gradients of the local model's loss with respect to its weights."""
        with tf.GradientTape() as tape:
            logits = self.local_model(images, training=True)
            loss_value = self.loss_fn(labels, logits)
        gradients = tape.gradient(loss_value, self.local_model.trainable_weights)
        return gradients

    def get_local_average(self):
        """Calculates the average gradients for the local model using the training dataset."""
        accumulated_gradients = [tf.zeros_like(w) for w in self.get_model_weights()]
        num_batches = 0

        for images, labels in self.train_dataset:
            gradients = self.get_gradients(images, labels)
            for i, gradient in enumerate(gradients):
                accumulated_gradients[i] = tf.add(accumulated_gradients[i], gradient)
            num_batches += 1

        if num_batches == 0:
            print(f"Warning: Client {self.id} has an empty training dataset. Returning zeros.")
            return [tf.zeros_like(w).numpy() for w in self.get_model_weights()]

        averaged_gradients = [tf.divide(grad, tf.cast(num_batches, tf.float32))
                              for grad in accumulated_gradients]

        return [grad.numpy() for grad in averaged_gradients]

    def _save_array_to_file(self, file_path, array):
        """Utility function to save a NumPy array to a binary file."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            with open(file_path, "wb") as f:
                array.astype(np.float32).tofile(f)
        except IOError as e:
            print(f"Error saving file {file_path}: {e}")

    def save_gradients_tofile(self, path):
        """Saves the average gradients to binary files in a directory specific to the client."""
        average_gradients = self.get_local_average()
        client_dir = os.path.join(path, str(self.id))

        for i, gradient_np in enumerate(average_gradients):
            file_name = ""
            if len(gradient_np.shape) == 1:
                file_name = f'bias{i+1}.bin'
            elif len(gradient_np.shape) == 2:
                file_name = f'dense{i+1}.bin'
            elif len(gradient_np.shape) == 4:
                file_name = f'conv{i+1}.bin'
            else:
                file_name = f'layer_{i+1}.bin' # Fallback name

            file_path = os.path.join(client_dir, file_name)
            self._save_array_to_file(file_path, gradient_np)

# Experimental Unit Class
class ExperimentalUnit:
    """
    Orchestrates the federated learning experiment, coordinating the central server and clients.
    """

    def __init__(self, num_clients):
        """
        Initializes the ExperimentalUnit.

        Args:
            num_clients: The number of clients in the experiment.
        """
        self.num_clients = num_clients

    def _calculate_images_seen_per_round(self, clients, batch_size):
        """Calculates the total number of images seen in a single round."""
        return sum(len(list(client.train_dataset)) * batch_size for client in clients)

    def _distribute_global_model(self, central_unit, clients):
        """Distributes the global model weights to all clients."""
        global_weights = central_unit.send_model()
        for client in clients:
            client.set_model_weights(global_weights)

    def _collect_and_aggregate_gradients(self, central_unit, clients, gradients_path):
        """Collects gradients from clients and aggregates them at the central server."""
        list_of_gradients = []
        for client in clients:
            grad = central_unit.get_update_from_user(client, gradients_path)
            if grad:
                list_of_gradients.append(grad)
        return central_unit.aggregate(list_of_gradients)

    def federated_training(self, central_unit, clients, test_dataset, gradients_path, rounds=10, batch_size = 32):
        """
        Performs federated training for a specified number of epochs.

        Args:
            central_unit: The CentralServer instance.
            clients: A list of Client instances.
            test_dataset: The test dataset for evaluation.
            gradients_path: The path to save/load client gradients.
            rounds: The number of training epochs.
            batch_size: The batch size used by the clients.

        Returns:
            A Pandas DataFrame containing the training metrics.
        """
        metrics = {}
        total_images_seen = 0
        images_seen_this_round = self._calculate_images_seen_per_round(clients, batch_size)

        for round in range(rounds):
            self._distribute_global_model(central_unit, clients)

            for client in clients:
                client.save_gradients_tofile(gradients_path)

            average_gradients = self._collect_and_aggregate_gradients(central_unit, clients, gradients_path)
            central_unit.apply_gradients(average_gradients)

            total_images_seen += images_seen_this_round

            loss, accuracy = central_unit.evaluate(test_dataset)
            metrics[round] = {
                'communication_round': round + 1,
                'loss': float(loss.numpy()),
                'accuracy': float(accuracy.numpy()),
                'images_seen_this_round': images_seen_this_round,
                'total_images_seen': total_images_seen,
            }

        return pd.DataFrame.from_dict(metrics, orient="index")

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
        print(f"Warning: max_dataset_size {max_dataset_size} is bigger than the train dataset size. setting it to train dataset size: {len(train_images)}")
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
        train_datasets.append(client_train_data.batch(batch_size, drop_remainder=True))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    return train_datasets, test_dataset

# Example Usage
if __name__ == "__main__":
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    gradients_path = os.path.join(output_dir, "federated_emulation_average")
    size, num_clients, rounds, batch_size = 2000, 10, 30, 32

    train_datasets, test_dataset = get_and_divide_dataset(num_clients, batch_size, size)
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    central_unit = CentralServer(cnn, optimizer, loss_function, metrics)
    central_unit.compile()

    clients = [
        Client(cnn, train_datasets[j], f"user00{j + 1}", optimizer, loss_function, metrics)
        for j in range(num_clients)
    ]

    experimental_unit = ExperimentalUnit(num_clients)
    metrics_df = experimental_unit.federated_training(central_unit, clients, test_dataset, gradients_path, rounds)

    metrics_df['round'] = metrics_df.index + 1
    output_csv_path = os.path.join(output_dir, f"federated_all_local_dataset_size_{size}.csv")
    try:
        metrics_df.to_csv(output_csv_path, index=False)
        print(f"Metrics saved to {output_csv_path}")
    except IOError as e:
        print(f"Error saving metrics to CSV: {e}")

    print(metrics_df)
