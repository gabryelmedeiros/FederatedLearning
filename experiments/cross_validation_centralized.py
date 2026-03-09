import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_and_process_dataset(max_size: int):
    """
    Load and preprocess the MNIST dataset.

    Args:
        max_size (int): Maximum number of samples to load from the dataset.

    Returns:
        Tuple of NumPy arrays: (x_train, y_train, x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:max_size] / 255.0
    y_train = y_train[:max_size]
    x_test = x_test[:max_size] / 255.0
    y_test = y_test[:max_size]

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


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


def train_model(model: tf.keras.Model, train_dataset: tf.data.Dataset, epochs: int = 10):
    """
    Train the provided CNN model on the given dataset.

    Args:
        model (tf.keras.Model): The CNN model to train.
        train_dataset (tf.data.Dataset): Training dataset.
        epochs (int): Number of epochs to train.

    Returns:
        pd.DataFrame: A DataFrame containing training metrics.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y, logits)
        batch_accuracy = train_acc_metric.result()
        train_acc_metric.reset_state()
        return loss_value, batch_accuracy

    data = {'Epoch': [], 'Batch': [], 'Images_Seen': [], 'Accuracy': [], 'Loss': []}
    total_images_seen = 0

    for epoch in range(epochs):
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value, batch_accuracy = train_step(x_batch, y_batch)
            total_images_seen += x_batch.shape[0]

            # Store batch-level data
            data['Epoch'].append(epoch + 1)
            data['Batch'].append(step + 1)
            data['Images_Seen'].append(total_images_seen)
            data['Accuracy'].append(float(batch_accuracy))
            data['Loss'].append(float(loss_value))

    return pd.DataFrame(data)


def evaluate_model(model: tf.keras.Model, x_val, y_val):
    """
    Evaluate a model on validation data.

    Args:
        model (tf.keras.Model): Trained model to evaluate
        x_val: Validation features
        y_val: Validation labels

    Returns:
        tuple: (accuracy, loss) on validation set
    """
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

    total_loss = 0
    num_batches = 0

    for x_batch, y_batch in val_dataset:
        predictions = model(x_batch, training=False)
        loss = loss_fn(y_batch, predictions)
        total_loss += float(loss)
        num_batches += 1
        test_acc_metric.update_state(y_batch, predictions)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return float(test_acc_metric.result()), avg_loss


def cross_validation_training(x_data, y_data, n_folds=5, batch_size=32, epochs=10):
    BYTES_PER_SAMPLE = 28 * 28 * 1 * 4  # 3136 bytes per sample (float32)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_histories = []
    fold_metrics = {
        'fold': [],
        'accuracy': [],
        'loss': []
    }

    round_metrics = {}

    fold_num = 1
    for train_idx, val_idx in kf.split(x_data):
        print(f"Training fold {fold_num}/{n_folds}")

        x_train_fold, x_val_fold = x_data[train_idx], x_data[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
        model = cnn()

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_fold, y_train_fold)).shuffle(1024).batch(batch_size)

        fold_history = train_model(model, train_dataset, epochs=epochs)
        fold_histories.append(fold_history)

        val_accuracy, val_loss = evaluate_model(model, x_val_fold, y_val_fold)

        fold_metrics['fold'].append(fold_num)
        fold_metrics['accuracy'].append(val_accuracy)
        fold_metrics['loss'].append(val_loss)

        for _, row in fold_history.iterrows():
            round_id = row['Images_Seen']
            if round_id not in round_metrics:
                round_metrics[round_id] = {
                    'accuracies': [],
                    'losses': [],
                    'samples': [],
                }
            round_metrics[round_id]['accuracies'].append(row['Accuracy'])
            round_metrics[round_id]['losses'].append(row['Loss'])
            round_metrics[round_id]['samples'].append(batch_size)

        fold_num += 1

    # Build final DataFrame
    rows = []
    for round_id in sorted(round_metrics.keys()):
        accs = round_metrics[round_id]['accuracies']
        losses = round_metrics[round_id]['losses']
        total_samples = sum(round_metrics[round_id]['samples'])
        bytes_transmitted = total_samples * BYTES_PER_SAMPLE

        rows.append({
            'round': round_id,
            'total_samples_processed': round_id,
            'accuracy': np.mean(accs),
            'accuracy_std': np.std(accs),
            'loss': np.mean(losses),
            'loss_std': np.std(losses),
            'total_bytes_transmitted': bytes_transmitted
        })

    df = pd.DataFrame(rows)
    fold_metrics_df = pd.DataFrame(fold_metrics)

    print(f"Cross-validation complete. Average validation accuracy: {fold_metrics_df['accuracy'].mean():.4f}")
    return df, fold_histories, fold_metrics_df



if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_and_process_dataset(2000)

    # Run cross-validation
    cv_metrics_df, fold_histories, fold_metrics_df = cross_validation_training(
        x_train, y_train, n_folds=5, batch_size=32, epochs=10
    )

    # Save cross-validation metrics for later plotting
    cv_metrics_df.to_csv('cv_metrics_batch_size_32_avg.csv', index=False)

    # Calculate and print summary statistics
    print("\nCross-validation results across folds:")
    print(f"Accuracy - Mean: {fold_metrics_df['accuracy'].mean():.4f}, Std: {fold_metrics_df['accuracy'].std():.4f}")
    print(f"Loss - Mean: {fold_metrics_df['loss'].mean():.4f}, Std: {fold_metrics_df['loss'].std():.4f}")

    print(cv_metrics_df.head())