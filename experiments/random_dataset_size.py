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
    output = tf.keras.layers.Dense(10, activation='softmax')(dense1) # Use layers.Dense

    model = tf.keras.models.Model(x_input, output)
    return model

class CentralServer:
    """
    Central server in a federated learning system.
    """
    def __init__(self, model_function, optimizer, loss, metrics):
        self.global_model = model_function()
        # Make sure optimizer, loss, and metrics are correctly instantiated
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
        """
        Aggregates gradients from multiple users.

        Args:
            user_gradients: A list where each element is a list of gradients
                            (as numpy arrays) for one user.

        Returns:
            A list of aggregated gradients (as numpy arrays), one for each layer.
        """
        if not user_gradients:
            print("Warning: No user gradients provided for aggregation. Returning zeros matching model weights.")
            # Return zeros matching the structure of the model's trainable weights
            return [tf.zeros_like(w).numpy() for w in self.global_model.trainable_weights]

        # Assume all users provide gradients for the same set of layers with matching shapes
        # Initialize aggregated gradients with the first user's gradients
        aggregated_gradients = [np.array(grad) for grad in user_gradients[0]]

        # Sum gradients from the remaining users
        for i in range(1, len(user_gradients)):
            user_grad_list = user_gradients[i]
            if len(user_grad_list) != len(aggregated_gradients):
                 print(f"Warning: User {i} gradient list length mismatch. Skipping user for aggregation.")
                 continue # Skip this user

            for j in range(len(aggregated_gradients)):
                grad = user_grad_list[j]
                if grad is None:
                    # If a gradient is None for a layer, skip it for this user
                    continue
                try:
                    # Ensure gradient shapes match before adding
                    if aggregated_gradients[j].shape != grad.shape:
                         print(f"Warning: Gradient shape mismatch for layer {j} from user {i}. Expected {aggregated_gradients[j].shape}, got {grad.shape}. Skipping layer for this user.")
                         continue # Skip this layer for this user
                    aggregated_gradients[j] += np.array(grad)
                except Exception as e:
                    print(f"Error aggregating gradient for layer {j} from user {i}: {e}. Skipping layer for this user.")
                    continue


        num_users = len(user_gradients) # Use the original count for averaging
        # Average the summed gradients
        # Handle potential division by zero if num_users was 0 (already checked)
        # or if some layers were skipped for all users.
        averaged_gradients = []
        for grad_sum in aggregated_gradients:
             # Check if grad_sum is still the initial state (only first user's grad)
             # A more robust check might be needed if initial state can be zero
             if num_users > 0:
                 averaged_gradients.append(grad_sum / num_users)
             else:
                 averaged_gradients.append(tf.zeros_like(grad_sum).numpy()) # Should not happen due to initial check


        return averaged_gradients


    @tf.function
    def apply_gradients(self, grads):
        """
        Applies the aggregated gradients to the global model.

        Args:
            grads: A list of gradients (as tf.Tensor) matching the model's trainable weights.
        """
        # Ensure gradients are tensors and match the structure/dtype of trainable weights
        trainable_weights = self.global_model.trainable_weights
        if len(grads) != len(trainable_weights):
            print("Error: Mismatch between number of aggregated gradients and trainable weights. Skipping gradient application.")
            return

        tensor_grads = []
        for i, grad in enumerate(grads):
             if grad is None:
                  # If an aggregated gradient is None (e.g., if all users had None for that layer), create a zero tensor
                  print(f"Warning: Aggregated gradient for layer {i} is None. Applying zero gradient.")
                  tensor_grads.append(tf.zeros_like(trainable_weights[i]))
             else:
                 try:
                    tensor_grads.append(tf.convert_to_tensor(grad, dtype=trainable_weights[i].dtype))
                    # Optional: check shape consistency here too if not already done in aggregate
                    # if tensor_grads[-1].shape != trainable_weights[i].shape:
                    #      print(f"Shape mismatch before applying gradient layer {i}. Expected {trainable_weights[i].shape}, got {tensor_grads[-1].shape}. Skipping.")
                    #      return
                 except Exception as e:
                      print(f"Error converting gradient {i} to tensor: {e}. Skipping gradient application.")
                      return # Exit the function on error


        # Apply gradients using the optimizer
        # The zip function pairs the gradients with the corresponding weights
        self.optimizer.apply_gradients(zip(tensor_grads, trainable_weights))


    def evaluate(self, test_dataset):
        # Use Keras model's evaluate method for simplicity and correctness
        # This method handles batching and metrics calculation internally
        # evaluate returns [loss, metric1, metric2, ...]
        results = self.global_model.evaluate(test_dataset, verbose=0)
        # Assuming 'accuracy' is the first metric after loss
        loss = results[0]
        accuracy = results[1] if len(results) > 1 else 0.0
        return loss, accuracy

class Client:
    """
    Represents a client in a federated learning system.
    """
    def __init__(self, function, train_dataset, id, optimizer, loss, metrics):
        self.local_model = function()
        self.train_dataset = train_dataset
        self.id = id
        # Store optimizer, loss, metrics (though not strictly used for local updates in this code)
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        # Instantiate the loss function
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # Use loss function instance

    def get_id(self):
        return self.id

    def compile(self):
         # Clients don't strictly need to compile in this gradient-based approach
         # as the central server handles optimization. Keeping it doesn't hurt.
        self.local_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)


    def set_model_weights(self, weights):
        try:
            self.local_model.set_weights(weights)
        except Exception as e:
            print(f"Error setting weights for client {self.id}: {e}")


    def get_model_weights(self):
        return self.local_model.get_weights()

    @tf.function
    def get_gradients(self, images, labels):
        """
        Computes gradients of the loss with respect to the model's trainable weights
        for a given batch of data.

        Args:
            images: A batch of images.
            labels: A batch of labels.

        Returns:
            A list of tf.Tensor objects representing the gradients.
            Returns None if gradients cannot be computed.
        """
        try:
            with tf.GradientTape() as tape:
                logits = self.local_model(images, training=True) # training=True is important for layers like Dropout, BatchNorm
                # Ensure loss calculation is correct (labels, predictions)
                # loss_fn handles softmax output correctly with from_logits=False
                loss_value = self.loss_fn(labels, logits)

            # Compute gradients
            gradients = tape.gradient(loss_value, self.local_model.trainable_weights)
            # Return the list of TF Tensors (or SymbolicTensors inside tf.function)
            return gradients
        except Exception as e:
            print(f"Error computing gradients in tf.function for client {self.id}: {e}")
            return None # Return None if computation fails


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
            test_datasets: Test datasets (validation sets) for each fold
            rounds: Maximum number of rounds to train
            batch_size: Batch size for training
            k_folds: Number of cross-validation folds
            sample_limit: Maximum number of samples to process in training

        Returns:
            DataFrame with averaged metrics across all folds, including standard deviation.
            Actual number of rounds completed.
        """
        all_fold_metrics = []

        # Store configuration information from the template
        optimizer_type = type(central_unit.optimizer)
        optimizer_config = central_unit.optimizer.get_config()
        loss_function = central_unit.loss
        metrics = central_unit.metrics

        for fold in range(k_folds):
            print(f"\n--- Cross-Validation Fold {fold + 1}/{k_folds} ---")

            # Create a fresh central unit for this fold with a fresh optimizer
            fresh_optimizer = optimizer_type.from_config(optimizer_config)
            fold_central_unit = CentralServer(cnn, fresh_optimizer, loss_function, metrics)
            fold_central_unit.compile() # Compile the fresh central model

            # Create fresh clients for this fold, each with a fresh model and optimizer
            clients = []
            for i in range(self.num_clients):
                 # Ensure client_train_datasets[i][fold] exists
                 if i < len(client_train_datasets) and fold < len(client_train_datasets[i]):
                    clients.append(
                        Client(
                            cnn,  # Each client gets a fresh model instance
                            client_train_datasets[i][fold], # Get the dataset for this client and this fold
                            f"client_{i + 1}_fold_{fold + 1}",
                            optimizer_type.from_config(optimizer_config),  # Fresh optimizer for each client
                            loss_function,
                            metrics
                        )
                    )
                 else:
                     print(f"Warning: Client {i} or Fold {fold+1} not found in client_train_datasets. Skipping client creation.")


            if not clients:
                 print(f"Error: No clients created for fold {fold + 1}. Skipping fold.")
                 continue


            # Clients don't strictly need compiling here for gradient exchange
            # but doing so sets up their internal optimizer/loss references.
            # for client in clients:
            #      client.compile() # Optional compile

            # Initialize metrics and counters for this fold
            fold_round_metrics = [] # Store metrics for each round in this fold
            total_samples_processed = 0
            round_num = 1

            # Create iterators for each client's dataset for this fold
            train_iterators = [iter(client.train_dataset) for client in clients]

            # Training loop
            # The loop runs up to the specified 'rounds' or until sample_limit is reached
            while total_samples_processed < sample_limit:
                print(f"Fold {fold + 1}/{k_folds}, Round {round_num} started.")

                # Distribute global weights from the fold's central unit to all clients
                global_weights = fold_central_unit.send_model()
                for client in clients:
                    client.set_model_weights(global_weights)

                # Collect gradients from one batch per active client
                round_gradients = [] # List to store gradient lists from each client
                active_clients_count = 0 # Count clients participating in this round's gradient exchange

                for client_index, client in enumerate(clients):
                    try:
                        if sum(1 for _ in client.train_dataset) == 0:
                            continue

                        # Get the next batch from the client's iterator
                        x_batch, y_batch = next(train_iterators[client_index])

                        # Get gradients from the client's model for this batch
                        # get_gradients now returns a list of tf.Tensors (or None on error)
                        gradients = client.get_gradients(x_batch, y_batch)

                        # Check if gradients were successfully computed
                        if gradients is not None and all(grad is not None for grad in gradients):
                             # Convert the list of tf.Tensors to a list of numpy arrays
                             numpy_gradients = [grad.numpy() for grad in gradients]
                             round_gradients.append(numpy_gradients)
                             active_clients_count += 1
                        else:
                             print(f"Warning: Client {client.id} failed to compute valid gradients for round {round_num}. Skipping client for aggregation in this round.")

                    except StopIteration:
                        # If a client runs out of data, re-initialize its iterator for the next round
                        print(f"Client {client.id} ran out of data for fold {fold+1}. Re-initializing iterator for future rounds.")
                        train_iterators[client_index] = iter(client.train_dataset)
                        # Client does not participate in this round if data wasn't available initially

                    except Exception as e:
                         print(f"An unexpected error occurred for client {client.id} in round {round_num}: {e}. Skipping client for aggregation.")


                # Aggregate gradients only if there are active clients that provided gradients
                if active_clients_count > 0:
                    # Aggregate the numpy gradients
                    average_grads_numpy = fold_central_unit.aggregate(round_gradients)

                    # Apply the aggregated gradients to the central model
                    # apply_gradients expects tf.Tensors, aggregate returns numpy, so conversion is needed
                    # apply_gradients method already handles conversion to tensor.
                    fold_central_unit.apply_gradients(average_grads_numpy)

                    # Update total samples processed based on the number of clients who provided gradients in this round
                    total_samples_processed += batch_size * active_clients_count
                else:
                    print(f"No active clients provided valid gradients in round {round_num} of fold {fold+1}. Skipping gradient application.")
                    # If no clients participated, do not count samples processed for this round

                # Evaluate the global model on the validation set for this fold
                # Ensure test_datasets[fold] exists
                if fold < len(test_datasets):
                    loss, accuracy = fold_central_unit.evaluate(test_datasets[fold])

                    # Store metrics for this round in this fold
                    fold_round_metrics.append({
                        'round': round_num,
                        'fold': fold + 1,
                        'total_samples_processed': total_samples_processed,
                        'accuracy': float(accuracy), # Ensure it's a standard float
                        'loss': float(loss)         # Ensure it's a standard float
                    })

                    print(f"  Fold {fold + 1}/{k_folds}, Round {round_num} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
                else:
                    print(f"Error: Validation dataset for fold {fold} not found. Skipping evaluation for this round.")
                    # Append placeholder or skip entirely for this round's metrics?
                    # Appending NaN might be informative:
                    # fold_round_metrics.append({
                    #     'round': round_num,
                    #     'fold': fold + 1,
                    #     'total_samples_processed': total_samples_processed,
                    #     'accuracy': np.nan,
                    #     'loss': np.nan
                    # })

                round_num += 1

            # Convert the list of round metrics for this fold into a DataFrame
            if fold_round_metrics: # Only create DataFrame if metrics were collected
                 fold_df = pd.DataFrame(fold_round_metrics)
                 all_fold_metrics.append(fold_df)
            else:
                 print(f"No metrics collected for fold {fold + 1}.")


        # Combine and average the metrics across all folds
        # Pass the list of per-fold dataframes to the revised averaging function
        # Handle case where all_fold_metrics is empty (e.g., if all folds were skipped)
        if all_fold_metrics:
            avg_metrics_df = self.average_metrics(all_fold_metrics=all_fold_metrics)
            # The actual number of rounds is the number of rows in the averaged DataFrame
            actual_rounds = len(avg_metrics_df)
        else:
            print("No fold metrics available for averaging.")
            avg_metrics_df = pd.DataFrame()
            actual_rounds = 0


        return avg_metrics_df, actual_rounds


    def average_metrics(self, all_fold_metrics):
        """
        Averages metrics across all folds and calculates standard deviation.

        Args:
            all_fold_metrics: A list of pandas DataFrames, one for each fold.
                              Each DataFrame is expected to have 'round', 'fold',
                              'total_samples_processed', 'accuracy', and 'loss' columns.

        Returns:
            DataFrame with averaged metrics and standard deviations per round.
        """
        if not all_fold_metrics:
            return pd.DataFrame()

        # Combine all fold dataframes into a single dataframe
        combined_df = pd.concat(all_fold_metrics, ignore_index=True)

        # Group by 'round' and calculate mean and standard deviation
        # Use .mean() to handle potential NaNs if some rounds/folds failed
        averaged_df = combined_df.groupby('round').agg(
            mean_accuracy=('accuracy', 'mean'),
            std_accuracy=('accuracy', 'std'),
            mean_loss=('loss', 'mean'),
            std_loss=('loss', 'std'),
            # We can also take the mean of total_samples_processed
            mean_total_samples_processed=('total_samples_processed', 'mean')
        ).reset_index() # reset_index turns the 'round' index back into a column

        # Rename columns for clarity
        averaged_df = averaged_df.rename(columns={
            'mean_accuracy': 'accuracy',
            'std_accuracy': 'accuracy_std',
            'mean_loss': 'loss',
            'std_loss': 'loss_std',
            'mean_total_samples_processed': 'total_samples_processed'
        })

        # Reorder columns to have 'round' and 'total_samples_processed' first
        cols = ['round', 'total_samples_processed', 'accuracy', 'accuracy_std', 'loss', 'loss_std']
        # Ensure all expected columns are present before selecting
        cols_present = [col for col in cols if col in averaged_df.columns]
        averaged_df = averaged_df[cols_present]


        return averaged_df

def get_and_divide_dataset(num_clients, batch_size, max_dataset_size, k_folds=5, normalization_factor=255.0, seed=42):
    """
    Loads the MNIST dataset, preprocesses it, and divides the training data among clients into k folds.
    Each client will get a random number of images (but total remains the same).
    """

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    if num_clients <= 0:
        raise ValueError("Number of clients must be greater than zero.")

    if max_dataset_size > len(train_images):
        print(f"Warning: max_dataset_size {max_dataset_size} is bigger than the train dataset size. Setting it to train dataset size: {len(train_images)}")
        max_dataset_size = len(train_images)

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
        test_datasets.append(
            tf.data.Dataset.from_tensor_slices((test_images[start:end], test_labels[start:end])).batch(batch_size)
        )

        # Training data for this fold (all except the validation fold)
        train_images_fold = np.concatenate((train_images[:start], train_images[end:]))
        train_labels_fold = np.concatenate((train_labels[:start], train_labels[end:]))

        # --- NEW: random distribution of samples among clients ---
        total_samples = len(train_images_fold)
        # Generate random proportions
        random_split = np.random.dirichlet(np.ones(num_clients), size=1)[0]
        samples_per_client = (random_split * total_samples).astype(int)

        # Ensure total matches exactly by adjusting the last client
        samples_per_client[-1] += total_samples - np.sum(samples_per_client)

        print(f"\nFold {k+1}: Images per client = {samples_per_client.tolist()} (Total = {np.sum(samples_per_client)})")

        # Assign random slices to each client
        idx = 0
        for i in range(num_clients):
            client_imgs = train_images_fold[idx:idx + samples_per_client[i]]
            client_lbls = train_labels_fold[idx:idx + samples_per_client[i]]
            idx += samples_per_client[i]

            client_train_data = tf.data.Dataset.from_tensor_slices((client_imgs, client_lbls)).batch(batch_size)
            client_train_datasets[i].append(client_train_data)

    return client_train_datasets, test_datasets



if __name__ == "__main__":
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    num_clients = 10
    batch_size = 21
    total_data_size = 2000 # This is the size of the training subset used from MNIST
    rounds = 30 # Max communication rounds
    k_folds = 5  # Number of cross-validation folds
    sample_limit = 8000 # Limit based on total samples processed across all clients and rounds

    # get_and_divide_dataset now returns client training data for each fold and validation data for each fold
    # max_dataset_size can be None to use the full MNIST training set
    client_train_datasets, validation_datasets = get_and_divide_dataset(
        num_clients=num_clients,
        batch_size=batch_size,
        max_dataset_size=total_data_size, # Use total_data_size as max_dataset_size for subsetting
        k_folds=k_folds
    )

    # Define optimizer, loss, and metrics
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    # The central_unit here is just used as a template to get optimizer/loss/metrics config
    central_unit_template = CentralServer(cnn, optimizer, loss_function, metrics)
    # Compiling the template is not strictly necessary if only config is needed

    experimental_unit = ExperimentalUnit(num_clients)

    # Pass the validation_datasets (which are the test sets for each fold)
    avg_metrics_df, actual_rounds = experimental_unit.federated_training(
        central_unit=central_unit_template, # Pass the template
        test_datasets=validation_datasets, # Pass the list of validation datasets for each fold
        client_train_datasets=client_train_datasets,
        rounds=rounds,
        batch_size=batch_size,
        k_folds=k_folds,
        sample_limit=sample_limit # Pass sample_limit
    )

    print("\n--- Average Metrics Across Folds ---")
    print(avg_metrics_df)

    # Use actual_rounds in the filename for accuracy
    output_csv_path = os.path.join(output_dir, f"federated_kfold_size_{total_data_size}_clients_{num_clients}_rounds_{actual_rounds}_batch_{batch_size}_kfold_{k_folds}_metrics.csv")
    try:
        if not avg_metrics_df.empty:
            avg_metrics_df.to_csv(output_csv_path, index=False)
            print(f"Average metrics saved to {output_csv_path}")
        else:
            print("No average metrics to save.")
    except IOError as e:
        print(f"Error saving metrics to CSV: {e}")
