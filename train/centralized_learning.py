import tensorflow as tf
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    # Example usage
    model = cnn()
    model.summary()

    x_train, y_train, x_test, y_test = load_and_process_dataset(2000)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    metrics_df = train_model(model, train_dataset, epochs=10)
    print(metrics_df.head())
