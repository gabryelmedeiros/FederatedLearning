import tensorflow as tf
import numpy as np
import os
import pandas as pd
from train.centralized_learning import cnn
from sklearn.model_selection import KFold

# Import the classes from your code
from train.simple_federated_emulation_per_average import CentralServer, Client, ExperimentalUnit, get_and_divide_dataset


def k_fold_cross_validation(num_clients, batch_size, dataset_size, k_folds=5, rounds=10,
                            output_dir="./outputs", result_dir="./results"):
    """
    Performs k-fold cross-validation for the federated learning setup.

    Args:
        num_clients: Number of clients to use in the federated learning.
        batch_size: Batch size for training.
        dataset_size: Size of the dataset to use.
        k_folds: Number of folds for cross-validation.
        rounds: Number of training rounds per fold.
        output_dir: Directory to save gradients.
        result_dir: Directory to save results.

    Returns:
        DataFrame with the consolidated results of all folds.
    """
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Limit dataset size if needed
    if dataset_size < len(train_images):
        train_images = train_images[:dataset_size]
        train_labels = train_labels[:dataset_size]
    else:
        dataset_size = len(train_images)
        print(f"Using full train dataset size: {dataset_size}")

    # Preprocess data
    train_images = train_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)

    test_images = test_images / 255.0
    test_images = np.expand_dims(test_images, axis=-1)

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Initialize K-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Storage for metrics across all folds
    all_fold_metrics = []

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"Starting fold {fold + 1}/{k_folds}")

        # Create fold specific directories
        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        gradients_path = os.path.join(fold_output_dir, "federated_emulation_average")

        # Split data for this fold
        fold_train_images, fold_train_labels = train_images[train_idx], train_labels[train_idx]
        fold_val_images, fold_val_labels = train_images[val_idx], train_labels[val_idx]

        # Create validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (fold_val_images, fold_val_labels)
        ).batch(batch_size)

        # Divide training data among clients
        client_datasets = []
        images_per_client = len(fold_train_images) // num_clients

        for i in range(num_clients):
            start_idx = i * images_per_client
            end_idx = start_idx + images_per_client

            # Handle last client potentially getting remaining data
            if i == num_clients - 1:
                end_idx = len(fold_train_images)

            client_images = fold_train_images[start_idx:end_idx]
            client_labels = fold_train_labels[start_idx:end_idx]

            client_dataset = tf.data.Dataset.from_tensor_slices(
                (client_images, client_labels)
            ).batch(batch_size, drop_remainder=True)

            client_datasets.append(client_dataset)

        # Create model components
        optimizer = tf.keras.optimizers.Adam()
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]

        # Create central server
        central_unit = CentralServer(cnn, optimizer, loss_function, metrics)
        central_unit.compile()

        # Create clients
        clients = [
            Client(cnn, client_datasets[j], f"user00{j + 1}", optimizer, loss_function, metrics)
            for j in range(num_clients)
        ]

        # Create experimental unit and run training
        experimental_unit = ExperimentalUnit(num_clients)
        fold_metrics = experimental_unit.federated_training(
            central_unit, clients, val_dataset, gradients_path, rounds, batch_size
        )

        # Add fold number to metrics
        fold_metrics['fold'] = fold + 1
        all_fold_metrics.append(fold_metrics)

        # Save fold-specific metrics
        fold_csv_path = os.path.join(result_dir, f"federated_fold_{fold + 1}_dataset_size_{dataset_size}.csv")
        try:
            fold_metrics.to_csv(fold_csv_path, index=False)
            print(f"Fold {fold + 1} metrics saved to {fold_csv_path}")
        except IOError as e:
            print(f"Error saving fold {fold + 1} metrics to CSV: {e}")

    # Concatenate all fold metrics
    all_metrics = pd.concat(all_fold_metrics)

    # Calculate averages across folds for each round
    avg_metrics = all_metrics.groupby('communication_round').agg({
        'loss': 'mean',
        'accuracy': 'mean',
        'images_seen_this_round': 'mean',
        'total_images_seen': 'mean'
    }).reset_index()

    avg_metrics['std_loss'] = all_metrics.groupby('communication_round')['loss'].std().values
    avg_metrics['std_accuracy'] = all_metrics.groupby('communication_round')['accuracy'].std().values

    # Save consolidated metrics
    consolidated_csv_path = os.path.join(result_dir, f"federated_cv_{k_folds}fold_dataset_size_{dataset_size}.csv")
    try:
        avg_metrics.to_csv(consolidated_csv_path, index=False)
        print(f"Consolidated metrics saved to {consolidated_csv_path}")
    except IOError as e:
        print(f"Error saving consolidated metrics to CSV: {e}")

    # Return the final metrics DataFrame
    return avg_metrics


# Example usage
if __name__ == "__main__":
    output_dir = "./outputs/cross_validation"
    result_dir = "./results/cross_validation"

    # Parameters
    size = 2000
    num_clients = 10
    rounds = 30
    batch_size = 3
    k_folds = 5

    # Run cross-validation
    avg_metrics = k_fold_cross_validation(
        num_clients=num_clients,
        batch_size=batch_size,
        dataset_size=size,
        k_folds=k_folds,
        rounds=rounds,
        output_dir=output_dir,
        result_dir=result_dir
    )
    avg_metrics.to_csv(os.path.join(result_dir, "federated_cv_results.csv"), index=False)
    print(f"Average metrics saved to {os.path.join(result_dir, 'federated_cv_results.csv')}")
    print("Cross-validation complete. Average metrics across all folds:")
    print(avg_metrics)
