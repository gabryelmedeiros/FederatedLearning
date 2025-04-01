import tensorflow as tf
import numpy as np
import os
import pandas as pd


# Central Server Class
class CentralServer:
    """
    Class CentralServer
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

    def get_update_from_user(self, user, path):
        def read_binary_file(file_path, shape):
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
            return data.reshape(shape)

        def read_gradients(path, gradients):
            gradients_read = []
            for i, gradient in enumerate(gradients):
                if len(gradient.shape) == 1:
                    gradient_read = read_binary_file(path + f'/bias{i+1}.bin', gradient.shape)
                elif len(gradient.shape) == 2:
                    gradient_read = read_binary_file(path + f'/dense{i+1}.bin', gradient.shape)
                elif len(gradient.shape) == 4:
                    gradient_read = read_binary_file(path + f'/conv{i+1}.bin', gradient.shape)
                gradients_read.append(gradient_read)
            return gradients_read

        return read_gradients(path + f"/{user.get_id()}", user.get_model_weights())

    def aggregate(self, user_gradients):
        average = [tf.zeros_like(w) for w in user_gradients[0]]
        for gradients in user_gradients:
            for i in range(len(average)):
                average[i] += gradients[i]

        for i in range(len(average)):
            average[i] /= len(user_gradients)

        return average

    @tf.function
    def apply_gradients(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))

    def evaluate(self, test_dataset):
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        batch_loss = 0.0

        @tf.function
        def evaluation_step(test_images, test_labels):
            predictions = self.global_model(test_images, training=False)
            batch_loss = self.loss(test_labels, predictions)
            accuracy_metric.update_state(test_labels, predictions)
            return batch_loss

        for test_images, test_labels in test_dataset:
            batch_loss = evaluation_step(test_images, test_labels)

        accuracy_result = accuracy_metric.result()
        return batch_loss, accuracy_result


# Client Class
class Client:
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

    def get_local_average(self):
        average = [tf.zeros_like(w) for w in self.get_model_weights()]
        batches = 0
        for step, (x_batch, y_batch) in enumerate(self.train_dataset):
            grad = self.get_gradients(x_batch, y_batch)
            for i, g in enumerate(grad):
                average[i] += tf.reshape(g, average[i].shape)
            batches += 1

        for i in range(len(average)):
            average[i] /= batches

        return average

    def save_gradients_tofile(self, path):
        def save_array(path, array):
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(path, "wb") as f:
                array.astype(np.float32).tofile(f)

        average_gradients = self.get_local_average()

        for i, gradient in enumerate(average_gradients):
            gradient_np = gradient.numpy()
            if len(gradient_np.shape) == 1:
                save_array(path + f'/{self.id}/bias{i+1}.bin', gradient_np)
            elif len(gradient_np.shape) == 2:
                save_array(path + f'/{self.id}/dense{i+1}.bin', gradient_np)
            elif len(gradient_np.shape) == 4:
                save_array(path + f'/{self.id}/conv{i+1}.bin', gradient_np)


class ExperimentalUnit:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def federated_training(self, central_unit, clients, test_dataset, path, epochs=10):
        metrics = {}
        total_images_seen = 0
        images_seen_this_round = 0

        for epoch in range(epochs):
            global_weights = central_unit.send_model()
            for client in clients:
                client.set_model_weights(global_weights)

            for client in clients:
                client.save_gradients_tofile(path)

            list_of_gradients = []
            for client in clients:
                grad = central_unit.get_update_from_user(client, path)
                if grad:
                    list_of_gradients.append(grad)

            average = central_unit.aggregate(list_of_gradients)
            central_unit.apply_gradients(average)

            images_seen_this_round = sum([
              len(list(client.train_dataset)) * 32
              for client in clients
            ])

            total_images_seen += images_seen_this_round

            loss, accuracy = central_unit.evaluate(test_dataset)
            metrics[epoch] = {
                'communication_round': epoch + 1,
                'loss': float(loss.numpy()),
                'accuracy': float(accuracy.numpy()),
                'images_seen_this_round': images_seen_this_round,
                'total_images_seen': total_images_seen
            }
        return pd.DataFrame.from_dict(metrics, orient="index")

def get_and_divide_dataset(num_clients, batch_size, max_size):
    # Loading the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Defining the maximum size of the dataset and the batch size
    MAX_SIZE = max_size
    BATCH_SIZE = batch_size

    # Slicing the dataset
    train_images = train_images[:MAX_SIZE]
    train_labels = train_labels[:MAX_SIZE]

    test_images = test_images[:MAX_SIZE]
    test_labels = test_labels[:MAX_SIZE]

    # Normalizing the dataset
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Expanding dimensions to add the channel dimension (since the model expects 28x28x1 input shape)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Dividing the training dataset among clients
    IMGS_PER_CLIENT = MAX_SIZE // num_clients
    train_datasets = []

    for i in range(num_clients):
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images[i * IMGS_PER_CLIENT:(i + 1) * IMGS_PER_CLIENT],
             train_labels[i * IMGS_PER_CLIENT:(i + 1) * IMGS_PER_CLIENT])
        )
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
        train_datasets.append(train_dataset)

    # Creating the test dataset to evaluate the global model
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_datasets, test_dataset

# Example Usage
if __name__ == "__main__":
    path = "./outputs/federated_emulation_average"
    size, num_clients, rounds, batch_size = 2000, 10, 30, 32

    def cnn():
        x_input = tf.keras.Input(shape=(28, 28, 1))
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x_input)
        pool1 = tf.keras.layers.MaxPool2D((3, 3))(conv1)
        flatten = tf.keras.layers.Flatten()(pool1)
        dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
        output = tf.keras.layers.Dense(10, activation='softmax')(dense1)
        return tf.keras.Model(inputs=x_input, outputs=output)

    train_datasets, test_dataset = get_and_divide_dataset(num_clients, batch_size, size)
    optimizer = tf.keras.optimizers.Adam()
    central_unit = CentralServer(cnn, optimizer, tf.keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
    central_unit.compile()

    clients = [
        Client(cnn, train_datasets[j], f"user00{j + 1}", optimizer, tf.keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
        for j in range(num_clients)
    ]

    experimental_unit = ExperimentalUnit(num_clients)
    metrics_df = experimental_unit.federated_training(central_unit, clients, test_dataset, path, rounds)

    metrics_df['epoch'] = metrics_df.index + 1
    metrics_df.to_csv(f"federated_all_local_dataset_size_{size}.csv", index=False)
    print(metrics_df)
