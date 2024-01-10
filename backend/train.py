from optimizers.adam import CustomAdamOptimizer
from datasets.load_mnist import load_mnist
from models.cnn_1 import cnn_1
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

import os

def train_cnn_1():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Load CNN model
    model = cnn_1()

    # Compile the model
    model.compile(optimizer=CustomAdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

    # Save the trained weights
    weights_dir = 'backend/weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    model.save_weights(os.path.join(weights_dir, 'cnn_1_weights.h5'))

if __name__ == "__main__":
    load_mnist()  # Visualize the dataset before training
    train_cnn_1()