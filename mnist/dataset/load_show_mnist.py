from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

def load_mnist():
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def visualize_mnist(train_images, train_labels):
    # Visualize the first 10 images and their labels
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i].squeeze(), cmap='gray')  # Squeeze removes the singleton dimension
        plt.title(f"Label: {train_labels[i].argmax()}")
        plt.axis('off')

    plt.savefig(os.path.join('static', 'images', 'dataset.png'))

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist()
    visualize_mnist(train_images, train_labels)
