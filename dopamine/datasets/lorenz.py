import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define the LorenzAttractorDataset class
class LorenzAttractorDataset(Dataset):
    def __init__(self, data, sequence_length, target_length):
        self.data = data
        self.sequence_length = sequence_length
        self.target_length = target_length

    def __len__(self):
        return len(self.data) - self.sequence_length - self.target_length + 1

    def __getitem__(self, index):
        # Get a sequence of data with the specified length
        sequence = self.data[index : index + self.sequence_length]
        # Get the target values for prediction
        target = self.data[
            index
            + self.sequence_length : index
            + self.sequence_length
            + self.target_length
        ]

        # Convert to PyTorch tensors
        sequence = torch.FloatTensor(sequence)
        target = torch.FloatTensor(target)

        return sequence, target

# Load and preprocess the Lorenz attractor dataset
def load_lorenz_attractor_dataset(data_length=5000):
    sigma, rho, beta = 10, 28, 8 / 3
    dt = 0.01
    x, y, z = [1.0], [0.0], [0.0]

    for i in range(1, data_length):
        dx = sigma * (y[i - 1] - x[i - 1])
        dy = x[i - 1] * (rho - z[i - 1]) - y[i - 1]
        dz = x[i - 1] * y[i - 1] - beta * z[i - 1]

        x.append(x[i - 1] + dx * dt)
        y.append(y[i - 1] + dy * dt)
        z.append(z[i - 1] + dz * dt)

    # Perform Min-Max scaling to normalize between -1 and 1
    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1
    z = (z - np.min(z)) / (np.max(z) - np.min(z)) * 2 - 1

    data = np.stack((x, y, z), axis=1)

    # Assuming you want to create a DataFrame with three columns 'x', 'y', and 'z'
    df = pd.DataFrame(data, columns=["x", "y", "z"])

    # Optionally, you can save the DataFrame to a CSV file
    # df.to_csv('lorenz_attractor_data.csv', index=False)

    # Return the entire DataFrame as a NumPy array
    return df.values

if __name__ == "__main__":
    # Create the dataset
    sequence_length = 499  # Change this to the desired sequence length
    target_length = 1
    data = load_lorenz_attractor_dataset(data_length=500)
    dataset = LorenzAttractorDataset(data, sequence_length, target_length)

    # Create a DataLoader to load a batch of data
    batch_size = 1  # Set batch size to 1 to visualize a single sequence
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of data
    for sequence_batch, target_batch in dataloader:
        # Convert PyTorch tensors back to NumPy arrays for plotting
        sequence_batch = sequence_batch.numpy()
        target_batch = target_batch.numpy()

        # Create a 3D scatter plot for the Lorenz attractor data
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection="3d")
        ax.scatter(
            sequence_batch[0, :, 0],
            sequence_batch[0, :, 1],
            sequence_batch[0, :, 2],
            label="Sequence",
            marker="o",
        )

        # Plot the target point
        ax.scatter(
            target_batch[0, :, 0],
            target_batch[0, :, 1],
            target_batch[0, :, 2],
            color="red",
            label="Target",
            marker="o",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Lorenz Attractor 3D Sequence and Target")
        ax.legend()
        plt.savefig(os.path.join('static', 'images', 'dataset.png'))
        # Break after processing the batch for demonstration
        break
