from optimizers.adam import CustomAdamOptimizer
from datasets.load_mnist import load_mnist
from datasets.load_fashion_mnist import load_fashion_mnist
from models.cnn_1 import cnn_1
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

def train_cnn_1():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

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
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Save the trained weights
    weights_dir = 'backend/weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    model.save_weights(os.path.join(weights_dir, 'cnn_1_weights.h5'))
    print("Model trained and weights saved")

    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()

    # Save the plot as "cnn_1.png"
    plt.savefig('static/images/cnn_1.png')

if __name__ == "__main__":
    train_cnn_1()
