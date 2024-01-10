# backend/datasets/load_fashion_mnist.py
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist  # Import the Fashion-MNIST dataset

def load_fashion_mnist():
    # Load Fashion-MNIST dataset
    (x_train, _), (_, _) = fashion_mnist.load_data()

    # Visualize the first 10 images
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_train[i], cmap='gray')
        ax.axis('off')

    # Save the visualization as fashion_dataset.png
    plt.savefig('static/images/dataset.png')

if __name__ == "__main__":
    load_fashion_mnist()