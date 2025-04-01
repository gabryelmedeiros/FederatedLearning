import tensorflow as tf
import numpy as np
import os
import pandas as pd


# Central Server Class
class CentralServer:
    def __init__(self, model_function, optimizer, loss, metrics):
        self.global_model = model_function()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compile(self):
        self.global_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def send_model(self):
        return self.global_model.get_weights()

    def get_update_from_user(self, user, path, num_batches):
        gradients_list = []
        for step in range(num_batches):
            try:
                def read_binary_file(file_path, shape):
                    with open(file_path, 'rb') as f:
                        data = np.fromfile(f, dtype=np.float32)
                    return data.reshape(shape)

                def read_gradients(path, gradients, batch_num):
                    gradients_read = []
                    for i, gradient in enumerate(gradients):
                        if len(gradient.shape) == 1:
                            gradient_read = read_binary_file(path + f'/batch_{batch_num}_bias{i+1}.bin', gradient.shape)
                        elif len(gradient.shape) == 2:
                            gradient_read = read_binary_file(path + f'/batch_{batch_num}_dense{i+1}.bin', gradient.shape)
                        elif len(gradient.shape) == 4:
                            gradient_read = read_binary_file(path + f'/batch_{batch_num}_conv{i+1}.bin', gradient.shape)
                        gradients_read.append(gradient_read)
                    return gradients_read

                gradients = read_gradients(path + f"/{user.get_id()}", user.get_model_weights(), step)
                gradients_list.append(gradients)
            except Exception as e:
                print(f"Error reading gradients from user {user.get_id()}, batch {step}: {e}")
                return None

        return gradients_list

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

    def save_array(self, path, array):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "wb") as f:
            array.astype(np.float32).tofile(f)

    def send_gradients_per_batch(self, path):
        for step, (x_batch, y_batch) in enumerate(self.train_dataset):
            gradients = self.get_gradients(x_batch, y_batch)
            for i, gradient in enumerate(gradients):
                gradient_np = gradient.numpy()
                if len(gradient_np.shape) == 1:
                    self.save_array(path + f'/{self.id}/batch_{step}_bias{i+1}.bin', gradient_np)
                elif len(gradient_np.shape) == 2:
                    self.save_array(path + f'/{self.id}/batch_{step}_dense{i+1}.bin', gradient_np)
                elif len(gradient_np.shape) == 4:
                    self.save_array(path + f'/{self.id}/batch_{step}_conv{i+1}.bin', gradient_np)


# Experimental Unit
class ExperimentalUnit:
    def __init__(self, num_clients, test_dataset):
        self.num_clients = num_clients
        self.test_dataset = test_dataset

    def federated_training(self, central_unit, clients, epochs, batch_size, path):
        metrics = {}
        cumulative_images_seen = []  # Para rastrear o total de imagens processadas
        accuracy_list = []  # Para rastrear a acurácia
        loss_list = []  # Para rastrear a perda

        print("Iniciando treinamento federado...\n")

        # Cria iteradores para os datasets de treinamento de cada cliente
        train_iterators = [iter(client.train_dataset) for client in clients]

        for epoch in range(epochs):
            print(f"Rodada {epoch + 1} iniciada.")

            # Envia os pesos globais para todos os clientes
            global_weights = central_unit.send_model()
            for client in clients:
                client.set_model_weights(global_weights)

            # Cada cliente processa apenas um mini-batch
            batch_gradients = []
            for client_index, client in enumerate(clients):
                try:
                    # Obtém o próximo mini-batch do cliente
                    x_batch, y_batch = next(train_iterators[client_index])
                except StopIteration:
                    # Se o dataset do cliente acabar, reinicia o iterador
                    train_iterators[client_index] = iter(client.train_dataset)
                    x_batch, y_batch = next(train_iterators[client_index])

                print(f"Cliente {client.get_id()} processando um mini-batch...")

                # Calcula os gradientes para o mini-batch atual
                gradients = client.get_gradients(x_batch, y_batch)
                batch_gradients.append(gradients)  # Adiciona os gradientes do cliente

            # Agrega os gradientes e atualiza o modelo global
            print("Agregando gradientes...")
            average = central_unit.aggregate(batch_gradients)
            print("Aplicando gradientes ao modelo global...")
            central_unit.apply_gradients(average)  # Atualiza o modelo global com os gradientes agregados

            # Atualiza o número de imagens processadas
            cumulative_images_seen.append((epoch + 1) * batch_size * len(clients))

            # Avalia o modelo global após cada rodada
            loss, accuracy = central_unit.evaluate(self.test_dataset)

            # Atualiza métricas
            accuracy_list.append(accuracy.numpy())
            loss_list.append(loss.numpy())

            metrics[epoch] = {'loss': loss.numpy(), 'accuracy': accuracy.numpy()}

            # Exibe métricas da rodada atual
            print(f"Métricas da Rodada {epoch + 1}:")
            print(f"  Loss: {loss.numpy():.4f}")
            print(f"  Acurácia: {accuracy.numpy():.4f}\n")

        # Dados para plotagem
        plot_data = {
            'images_seen': cumulative_images_seen,
            'accuracy': accuracy_list,
            'loss': loss_list
        }

        df = pd.DataFrame(data=plot_data)

        return df

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
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_datasets.append(train_dataset)

    # Creating the test dataset to evaluate the global model
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_datasets, test_dataset

# Example usage
if __name__ == "__main__":
    path = "./outputs/federated_emulation_per_batch"

    def cnn():
        x_input = tf.keras.Input(shape=(28, 28, 1))
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x_input)
        pool1 = tf.keras.layers.MaxPool2D((3, 3))(conv1)
        flatten = tf.keras.layers.Flatten()(pool1)
        dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
        output = tf.keras.layers.Dense(10, activation='softmax')(dense1)
        return tf.keras.Model(inputs=x_input, outputs=output)

    num_clients = 5
    train_datasets, test_dataset = get_and_divide_dataset(num_clients, 3, 2000)

    optimizer = tf.keras.optimizers.Adam()
    central_unit = CentralServer(cnn, optimizer, tf.keras.losses.SparseCategoricalCrossentropy(), ["accuracy"])
    central_unit.compile()

    clients = [Client(cnn, train_datasets[i], f"client_{i+1}", optimizer, tf.keras.losses.SparseCategoricalCrossentropy(), ["accuracy"]) for i in range(num_clients)]

    experimental_unit = ExperimentalUnit(num_clients, test_dataset)
    metrics_df = experimental_unit.federated_training(central_unit, clients, 30, 3, path)
    print(metrics_df)
