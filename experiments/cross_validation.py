import tensorflow as tf
import numpy as np
import os
import pandas as pd

def cnn() -> tf.keras.Model:
    """
    Create a small CNN model for image classification.

    Returns:
        tf.keras.Model: A compiled CNN model.
    """
    x_input = tf.keras.Input(shape=(28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x_input)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(conv1)
    flatten = tf.keras.layers.Flatten()(pool1)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    output = tf.keras.layers.Dense(10, activation='softmax')(dense1)

    model = tf.keras.models.Model(x_input, output)
    return model

class CentralServer:
    """
    Central server in a federated learning system.
    """
    def __init__(self, model_function, optimizer, loss, metrics):
        self.global_model = model_function()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compile(self):
        self.global_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def send_model(self):
        return self.global_model.get_weights()

    def set_weights(self, weights):
        self.global_model.set_weights(weights)

    def aggregate(self, user_gradients):
        if not user_gradients:
            print("Warning: No user gradients provided for aggregation. Returning zeros.")
            return [tf.zeros_like(w).numpy() for w in self.global_model.trainable_weights]

        aggregated_gradients = [np.array(grad) for grad in user_gradients[0]]
        for i in range(1, len(user_gradients)):
            for j in range(len(aggregated_gradients)):
                aggregated_gradients[j] += np.array(user_gradients[i][j])

        num_users = len(user_gradients)
        averaged_gradients = [grad / num_users for grad in aggregated_gradients]
        return averaged_gradients

    @tf.function
    def apply_gradients(self, grads):
        tensor_grads = [tf.convert_to_tensor(g) for g in grads]
        self.optimizer.apply_gradients(zip(tensor_grads, self.global_model.trainable_weights))

    def evaluate(self, test_dataset):
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

class Client:
    """
    Represents a client in a federated learning system.
    """
    def __init__(self, function, train_dataset, id, optimizer, loss, metrics):
        self.local_model = function()
        self.train_dataset = train_dataset
        self.id = id
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def get_id(self):
        return self.id

    def compile(self):
        self.local_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def set_model_weights(self, weights):
        self.local_model.set_weights(weights)

    def get_model_weights(self):
        return self.local_model.get_weights()

    @tf.function
    def get_gradients(self, images, labels):
        with tf.GradientTape() as tape:
            logits = self.local_model(images, training=True)
            loss_value = self.loss_fn(labels, logits)
        gradients = tape.gradient(loss_value, self.local_model.trainable_weights)
        return gradients

class ExperimentalUnit:
    """
    Orchestrates the federated learning experiment with k-fold cross-validation.
    """
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def federated_training(self, central_unit, client_train_datasets, test_datasets, rounds=10, batch_size=32,
                           k_folds=5,
                           sample_limit=8000):
        """
        Performs federated training with k-fold cross-validation.

        Args:
            central_unit: The original central server (only used as a template)
            client_train_datasets: Training datasets for each client across k folds
            test_datasets: Test datasets for each fold
            rounds: Maximum number of rounds to train
            batch_size: Batch size for training
            k_folds: Number of cross-validation folds
            sample_limit: Maximum number of samples to process in training

        Returns:
            DataFrame with averaged metrics across all folds
        """
        all_fold_metrics = []

        # Store configuration information
        optimizer_type = type(central_unit.optimizer)
        optimizer_config = central_unit.optimizer.get_config()
        loss_function = central_unit.loss
        metrics = central_unit.metrics

        for fold in range(k_folds):
            print(f"\n--- Cross-Validation Fold {fold + 1}/{k_folds} ---")

            # Create a fresh central unit for this fold
            fresh_optimizer = optimizer_type.from_config(optimizer_config)
            fold_central_unit = CentralServer(cnn, fresh_optimizer, loss_function, metrics)
            fold_central_unit.compile()

            # Create fresh clients for this fold
            clients = [
                Client(
                    cnn,  # Each client gets a fresh model
                    client_train_datasets[i][fold],
                    f"client_{i + 1}_fold_{fold + 1}",
                    optimizer_type.from_config(optimizer_config),  # Fresh optimizer for each client
                    loss_function,
                    metrics
                ) for i in range(self.num_clients)
            ]

            for client in clients:
                client.compile()

            # Initialize metrics and counters for this fold
            fold_metrics = {}
            total_samples_processed = 0
            round_num = 1
            accuracy_list = []
            loss_list = []

            train_iterators = [iter(client.train_dataset) for client in clients]

            # Training loop
            while total_samples_processed < sample_limit:
                print(f"Round {round_num} started.")

                # Distribute global weights from the fold's central unit
                global_weights = fold_central_unit.send_model()
                for client in clients:
                    client.set_model_weights(global_weights)

                # Collect and aggregate gradients
                batch_gradients = []
                for client_index, client in enumerate(clients):
                    try:
                        x_batch, y_batch = next(train_iterators[client_index])
                    except StopIteration:
                        train_iterators[client_index] = iter(client.train_dataset)
                        x_batch, y_batch = next(train_iterators[client_index])

                    gradients = client.get_gradients(x_batch, y_batch)
                    batch_gradients.append(gradients)

                average = fold_central_unit.aggregate(batch_gradients)
                fold_central_unit.apply_gradients(average)

                total_samples_processed += batch_size * len(clients)

                # Evaluate on the validation set for this fold
                loss, accuracy = fold_central_unit.evaluate(test_datasets[fold])

                accuracy_list.append(accuracy.numpy())
                loss_list.append(loss.numpy())
                fold_metrics[round_num] = {'loss': loss.numpy(), 'accuracy': accuracy.numpy()}

                print(f"  Loss: {loss.numpy():.4f}")
                print(f"  Accuracy: {accuracy.numpy():.4f}\n")
                round_num += 1

            # Store metrics for this fold, using the actual number of rounds completed
            actual_rounds = len(accuracy_list)
            fold_data = {
                'total_samples_processed': [i * batch_size * len(clients) for i in range(1, actual_rounds + 1)],
                'accuracy': accuracy_list,
                'loss': loss_list
            }
            fold_df = pd.DataFrame(data=fold_data)
            all_fold_metrics.append(fold_df)

        # Average the metrics across all folds
        avg_metrics = self.average_metrics(all_fold_metrics=all_fold_metrics)
        return avg_metrics, actual_rounds


    def average_metrics(self, all_fold_metrics):
            """Averages metrics across all folds."""

            if not all_fold_metrics:
                return pd.DataFrame()

            avg_metrics = all_fold_metrics[0].copy()

            for fold_metrics in all_fold_metrics[1:]:
                avg_metrics['accuracy'] += fold_metrics['accuracy']
                avg_metrics['loss'] += fold_metrics['loss']
                if 'total_samples_processed' in avg_metrics:
                    avg_metrics['total_samples_processed'] = avg_metrics['total_samples_processed']

            avg_metrics['accuracy'] /= len(all_fold_metrics)
            avg_metrics['loss'] /= len(all_fold_metrics)

            return avg_metrics

def get_and_divide_dataset(num_clients, batch_size, max_dataset_size, k_folds=5, normalization_factor=255.0, seed=42):
    """
    Loads the MNIST dataset, preprocesses it, and divides the training data among clients into k folds.
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

    # Shuffle the data
    np.random.seed(seed)
    indices = np.random.permutation(len(train_images))
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    # Calculate fold sizes
    fold_size = len(train_images) // k_folds
    if fold_size == 0:
        raise ValueError("max_dataset_size is too small for the number of folds.")

    # Create k folds for training and testing
    client_train_datasets = [[] for _ in range(num_clients)]
    test_datasets = []

    for k in range(k_folds):
        start, end = k * fold_size, (k + 1) * fold_size

        # Validation set for this fold
        val_images_fold = train_images[start:end]
        val_labels_fold = train_labels[start:end]
        test_datasets.append(tf.data.Dataset.from_tensor_slices((test_images[start:end], test_labels[start:end])).batch(batch_size))

        # Training data for this fold (all except the validation fold)
        train_images_fold = np.concatenate((train_images[:start], train_images[end:]))
        train_labels_fold = np.concatenate((train_labels[:start], train_labels[end:]))

        # Distribute training data among clients
        images_per_client = len(train_images_fold) // num_clients
        for i in range(num_clients):
            client_train_data = tf.data.Dataset.from_tensor_slices(
                (train_images_fold[i * images_per_client:(i + 1) * images_per_client],
                 train_labels_fold[i * images_per_client:(i + 1) * images_per_client])
            ).batch(batch_size)
            client_train_datasets[i].append(client_train_data)

    return client_train_datasets, test_datasets

if __name__ == "__main__":
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    gradients_path = os.path.join(output_dir, "federated_emulation")

    num_clients = 10
    batch_size = 24
    total_data_size = 2000
    rounds = 30
    k_folds = 5  # Set the number of folds for cross-validation

    client_train_datasets, test_datasets = get_and_divide_dataset(num_clients, batch_size, total_data_size, k_folds=k_folds)

    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    central_unit = CentralServer(cnn, optimizer, loss_function, metrics)
    central_unit.compile()

    experimental_unit = ExperimentalUnit(num_clients)
    avg_metrics_df, actual_rounds = experimental_unit.federated_training(
        central_unit=central_unit,
        test_datasets=test_datasets,
        client_train_datasets=client_train_datasets,
        rounds=rounds,
        batch_size=batch_size,
        k_folds=k_folds)

    print("\n--- Average Metrics Across Folds ---")
    print(avg_metrics_df)

    output_csv_path = os.path.join(output_dir, f"federated_kfold_size_{total_data_size}_clients_{num_clients}_rounds_{actual_rounds}_batch_{batch_size}_kfold_{k_folds}.csv")
    try:
        avg_metrics_df.to_csv(output_csv_path, index=False)
        print(f"Average metrics saved to {output_csv_path}")
    except IOError as e:
        print(f"Error saving metrics to CSV: {e}")