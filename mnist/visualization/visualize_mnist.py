# visualization_utils.py
import matplotlib.pyplot as plt
import os
from mnist.dataset.load_show_mnist import load_mnist

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