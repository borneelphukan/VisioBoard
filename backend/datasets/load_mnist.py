# backend/datasets/load_mnist.py
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_mnist():
    # Load MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()

    # Visualize the first 10 images
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_train[i], cmap='gray')
        ax.axis('off')

    # Save the visualization as dataset.png
    plt.savefig('static/images/dataset.png')

if __name__ == "__main__":
    load_mnist()
