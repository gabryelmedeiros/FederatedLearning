import tensorflow as tf
import numpy as np
import os
import pandas as pd
from train.centralized_learning import cnn # Assuming this imports your CNN model function
from sklearn.model_selection import KFold

# Import the classes from your file
# ASSUMPTION: These classes implement the gradient sharing logic previously discussed.
# ASSUMPTION: ExperimentalUnit.federated_training(central_unit, clients, val_dataset, gradients_path, rounds, batch_size)
#             when called for a single fold, returns a DataFrame for that fold with columns:
#             'round', 'samples_this_round', 'accuracy', 'loss'.
#             ('samples_this_round' is the number of samples processed in that specific round,
#              'accuracy' and 'loss' are evaluated on the val_dataset.)
from train.simple_federated_emulation_per_average import CentralServer, Client, ExperimentalUnit, get_and_divide_dataset


def k_fold_cross_validation(num_clients, batch_size, dataset_size, k_folds=5, rounds=30, # Increased default rounds
                            output_dir="./outputs", result_dir="./results"):
    """
    Performs k-fold cross-validation for the federated learning setup.

    Args:
        num_clients: Number of clients to use in the federated learning.
        batch_size: Batch size for training.
        dataset_size: Size of the dataset subset to use from MNIST train data.
        k_folds: Number of folds for cross-validation.
        rounds: Number of training rounds per fold.
        output_dir: Directory to save gradients (used by ExperimentalUnit).
        result_dir: Directory to save results (per fold and consolidated).

    Returns:
        DataFrame with the consolidated averaged results and standard deviations
        across all folds, with columns matching the plotting script requirements.
    """
    # Load MNIST dataset (only train split used for CV)
    (train_images, train_labels), (test_images, test_labels_orig) = tf.keras.datasets.mnist.load_data() # Load test_images_orig just in case

    # Limit dataset size if needed, or use full dataset
    if dataset_size is not None and dataset_size < len(train_images):
        train_images = train_images[:dataset_size]
        train_labels = train_labels[:dataset_size]
    else:
        dataset_size = len(train_images) # Use full size if dataset_size is None or larger than data
        print(f"Using full train dataset size for cross-validation: {dataset_size}")


    # Preprocess train data for CV (normalize and expand dimensions)
    train_images = train_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)

    # Note: The original test_images are loaded but not used within the KFold CV
    # process here, as evaluation in each fold is done on the validation split
    # derived from the training data.

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Initialize K-fold cross-validation on the *indices* of the training data
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Storage for metrics DataFrames from each fold
    all_fold_raw_metrics = []

    # Perform k-fold cross-validation
    # kf.split gives indices into the train_images/train_labels arrays
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"Starting fold {fold_idx + 1}/{k_folds}")

        # Create fold specific output directory for gradients etc.
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        gradients_path = os.path.join(fold_output_dir, "federated_emulation_average") # Path for gradients/logs if used by ExperimentalUnit


        # --- Data Preparation for the Current Fold ---
        # Split data for this fold using the indices from KFold
        fold_train_images, fold_train_labels = train_images[train_idx], train_labels[train_idx]
        fold_val_images, fold_val_labels = train_images[val_idx], train_labels[val_idx]

        # Create validation dataset for this fold (used for evaluation per round)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (fold_val_images, fold_val_labels)
        ).batch(batch_size) # Batch size for evaluation

        # Divide the training split of this fold among clients
        # We need to ensure clients have data for batching. drop_remainder=True helps keep batch sizes consistent.
        # Need to check if data per client is sufficient for at least one batch.
        total_fold_train_size = len(fold_train_images)
        images_per_client_in_fold_train = total_fold_train_size // num_clients
        if images_per_client_in_fold_train < batch_size:
             print(f"Warning: In fold {fold_idx + 1}, training data per client ({images_per_client_in_fold_train}) is less than batch size ({batch_size}). Clients may have empty or incomplete batches. Consider increasing dataset_size, decreasing num_clients, or decreasing batch_size.")


        client_datasets = []
        for i in range(num_clients):
            start_idx = i * images_per_client_in_fold_train
            end_idx = start_idx + images_per_client_in_fold_train

            # Handle last client potentially getting remaining data (if drop_remainder=False)
            # With drop_remainder=True, we just distribute full batches
            if i == num_clients - 1:
                 client_images = fold_train_images[start_idx:]
                 client_labels = fold_train_labels[start_idx:]
            else:
                 client_images = fold_train_images[start_idx:end_idx]
                 client_labels = fold_train_labels[start_idx:end_idx]


            # Ensure client dataset is batched and potentially shuffled/repeated
            # drop_remainder=True ensures consistent batch size, necessary for some aggregation methods
            client_dataset = tf.data.Dataset.from_tensor_slices(
                (client_images, client_labels)
            ).shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).repeat() # Repeat for multiple rounds


            client_datasets.append(client_dataset)

        # --- Model and Training Setup for the Current Fold ---
        # Re-initialize model, optimizer for each fold to ensure independent training
        optimizer = tf.keras.optimizers.Adam()
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]

        # Create central server for this fold
        central_unit = CentralServer(cnn, optimizer, loss_function, metrics)
        central_unit.compile() # Compile the central model for this fold

        # Create clients for this fold
        clients = [
            Client(cnn, client_datasets[j], f"client_{j + 1}_fold_{fold_idx + 1}", optimizer, loss_function, metrics)
            for j in range(num_clients)
        ]
        # Clients might need compiling depending on their internal implementation,
        # but often not necessary if they only compute gradients.


        # --- Run Federated Training for the Current Fold ---
        # Assuming ExperimentalUnit is designed to train for a single fold and return its metrics
        experimental_unit = ExperimentalUnit(num_clients) # Re-instantiate if needed, or reuse
        print(f"Running federated training for fold {fold_idx + 1}...")

        # Call the ExperimentalUnit's training method
        # ASSUMED RETURN: DataFrame for this fold with columns 'round', 'samples_this_round', 'accuracy', 'loss'
        fold_metrics_df = experimental_unit.federated_training(
            central_unit, clients, val_dataset, gradients_path, rounds, batch_size # Pass parameters needed by your ExperimentalUnit
        )

        # --- Process Metrics from the Current Fold ---
        if fold_metrics_df is not None and not fold_metrics_df.empty:
            # Calculate cumulative samples processed for this fold
            # Check if 'samples_this_round' exists, if not, try to infer or skip
            if 'samples_this_round' in fold_metrics_df.columns:
                 fold_metrics_df['total_samples_processed'] = fold_metrics_df['samples_this_round'].cumsum()
            elif 'images_seen_this_round' in fold_metrics_df.columns: # Check for alternative name
                 print(f"Using 'images_seen_this_round' as samples per round for fold {fold_idx + 1}.")
                 fold_metrics_df['total_samples_processed'] = fold_metrics_df['images_seen_this_round'].cumsum()
                 # Rename if needed to match expected structure for later averaging
                 fold_metrics_df = fold_metrics_df.rename(columns={'images_seen_this_round': 'samples_this_round'})
            else:
                 print(f"Warning: Cannot find 'samples_this_round' or 'images_seen_this_round' in fold {fold_idx + 1} metrics. Cannot calculate 'total_samples_processed'. Skipping fold metrics.")
                 continue # Skip this fold if cumulative samples can't be calculated


            # Add fold number to the DataFrame
            fold_metrics_df['fold'] = fold_idx + 1

            # Select and rename columns to a consistent format before concatenation
            # Ensure the columns expected by the averaging step exist
            required_cols_for_averaging = ['round', 'total_samples_processed', 'accuracy', 'loss', 'fold']
            if all(col in fold_metrics_df.columns for col in required_cols_for_averaging):
                all_fold_raw_metrics.append(fold_metrics_df[required_cols_for_averaging])
                print(f"Fold {fold_idx + 1} metrics processed and added.")
            else:
                 print(f"Warning: Fold {fold_idx + 1} metrics DataFrame is missing required columns for averaging: {required_cols_for_averaging}. Skipping fold metrics.")


        else:
            print(f"ExperimentalUnit.federated_training for fold {fold_idx + 1} returned empty or None metrics.")


        # Save fold-specific raw metrics (optional but good for debugging)
        fold_csv_path = os.path.join(result_dir, f"federated_raw_fold_{fold_idx + 1}_dataset_size_{dataset_size}.csv")
        try:
             if fold_metrics_df is not None and not fold_metrics_df.empty:
                 fold_metrics_df.to_csv(fold_csv_path, index=False)
                 print(f"Fold {fold_idx + 1} raw metrics saved to {fold_csv_path}")
        except IOError as e:
            print(f"Error saving fold {fold_idx + 1} raw metrics to CSV: {e}")


    # --- Consolidate and Average Metrics Across All Folds ---
    if not all_fold_raw_metrics:
        print("No valid fold metrics collected. Cannot consolidate.")
        return pd.DataFrame() # Return empty DataFrame

    # Concatenate all raw fold metrics DataFrames
    combined_raw_metrics = pd.concat(all_fold_raw_metrics, ignore_index=True)

    # Group by 'round' and calculate the mean and standard deviation
    # Ensure 'total_samples_processed' is also averaged per round
    avg_metrics = combined_raw_metrics.groupby('round').agg(
        mean_total_samples_processed=('total_samples_processed', 'mean'),
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        mean_loss=('loss', 'mean'),
        std_loss=('loss', 'std')
    ).reset_index() # reset_index turns the 'round' index back into a column

    # Rename columns to match the desired format for plotting
    final_columns = {
        'round': 'round', # Already named correctly
        'mean_total_samples_processed': 'total_samples_processed',
        'mean_accuracy': 'accuracy',
        'std_accuracy': 'accuracy_std',
        'mean_loss': 'loss',
        'std_loss': 'loss_std'
    }
    # Select and rename only the columns we need in the final output
    avg_metrics = avg_metrics.rename(columns=final_columns)

    # Ensure the final DataFrame has exactly the expected columns in order
    expected_final_cols = ['round', 'total_samples_processed', 'accuracy', 'accuracy_std', 'loss', 'loss_std']
    # Filter for columns that actually exist after aggregation
    final_cols_present = [col for col in expected_final_cols if col in avg_metrics.columns]
    avg_metrics = avg_metrics[final_cols_present]


    # Save consolidated metrics (matching the plotting input format)
    consolidated_csv_path = os.path.join(result_dir, f"federated_cv_{k_folds}fold_dataset_size_{dataset_size}_averaged_metrics.csv")
    try:
        avg_metrics.to_csv(consolidated_csv_path, index=False)
        print(f"\nConsolidated averaged metrics saved to {consolidated_csv_path}")
    except IOError as e:
        print(f"Error saving consolidated averaged metrics to CSV: {e}")

    # Return the final averaged metrics DataFrame
    return avg_metrics


# Example usage
if __name__ == "__main__":
    # This part demonstrates how to run the k-fold CV and get the final DataFrame
    # You would then load this CSV (or use the returned DataFrame directly)
    # in your separate plotting script/notebook cells.
    output_dir = "./outputs/cross_validation_separate"
    result_dir = "./results/cross_validation_separate"

    # Parameters for the CV run
    size = 2000 # Dataset subset size
    num_clients = 10
    rounds = 30 # Rounds per fold
    batch_size = 32
    k_folds = 5

    print("Starting k-fold cross-validation...")
    final_averaged_metrics_df = k_fold_cross_validation(
        num_clients=num_clients,
        batch_size=batch_size,
        dataset_size=size,
        k_folds=k_folds,
        rounds=rounds,
        output_dir=output_dir,
        result_dir=result_dir
    )

    print("\n--- Final Averaged Metrics Across Folds ---")
    print(final_averaged_metrics_df)

    # The 'final_averaged_metrics_df' DataFrame now has the desired structure
    # ('round', 'total_samples_processed', 'accuracy', 'accuracy_std', 'loss', 'loss_std')
    # You can save this DataFrame to a CSV and then load it in your plotting script,
    # or pass this DataFrame directly to your plotting function if running in the same session.
    # The save step inside k_fold_cross_validation already saves it.