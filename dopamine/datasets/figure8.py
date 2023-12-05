import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the Figure8Dataset class
class Figure8Dataset(Dataset):
    def __init__(self, data, sequence_length, target_length):
        self.data = data
        self.sequence_length = sequence_length
        self.target_length = target_length

    def __len__(self):
        return len(self.data) - self.sequence_length - self.target_length + 1

    def __getitem__(self, index):
        # Get a sequence of data with the specified length
        sequence = self.data[index:index + self.sequence_length]
        # Get the target values for prediction
        target = self.data[index + self.sequence_length:index + self.sequence_length + self.target_length]

        # Convert to PyTorch tensors
        sequence = torch.FloatTensor(sequence)
        target = torch.FloatTensor(target)

        return sequence, target

# Load and preprocess the Figure-8 dataset
def load_figure8_dataset(data_length=500):
    time_steps = np.linspace(0, 2 * np.pi, data_length + 1)
    x = np.sin(time_steps)
    y = np.sin(time_steps) * np.cos(time_steps)
    data = np.stack((x, y), axis=1)

    # Assuming you want to create a DataFrame with two columns 'x' and 'y'
    df = pd.DataFrame(data, columns=['x', 'y'])

    # Optionally, you can save the DataFrame to a CSV file
    # df.to_csv('figure8_data.csv', index=False)

    # Return the entire DataFrame as a NumPy array
    return df.values

if __name__ == '__main__':
    # Create the dataset
    sequence_length = 50
    target_length = 1
    data = load_figure8_dataset()
    dataset = Figure8Dataset(data, sequence_length, target_length)

    # Create a DataLoader to load a batch of data
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of data
    for sequence_batch, target_batch in dataloader:
        # Convert PyTorch tensors back to NumPy arrays for plotting
        sequence_batch = sequence_batch.numpy()
        target_batch = target_batch.numpy()

        # Create a scatter plot for x and y features with the target point
        plt.figure(figsize=(8, 6))
        plt.scatter(sequence_batch[0, :, 0], sequence_batch[0, :, 1], label='Sequence', marker='o', s=50)
        plt.scatter(target_batch[0, :, 0], target_batch[0, :, 1], label='Target', marker='x', color='red', s=100)

        plt.xlabel('Feature x')
        plt.ylabel('Feature y')
        plt.title('Single Sequence and Target')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join('static', 'images', 'figure8_plot.png'))

        # Break after processing the batch for demonstration
        break