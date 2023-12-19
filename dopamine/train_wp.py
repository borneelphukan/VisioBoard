import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

from models.rnn import RNNModel
from datasets.figure8 import Figure8Dataset, load_figure8_dataset
from datasets.lorenz import LorenzAttractorDataset, load_lorenz_attractor_dataset
from datasets.rossler import RosslerAttractorDataset, load_rossler_attractor_dataset

from wp import WeightPerturbation
from optimizer import Dopamine
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

torch.set_num_threads(4)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training in {device}")

def test_rossler(model, data_length,batch_size, sequence_length,target_length,shuffle):
    
    # Load and preprocess the Rossler Attractor dataset
    test_data = load_rossler_attractor_dataset(data_length)

    # Create a DataLoader for the dataset
    test_dataset = RosslerAttractorDataset(test_data, sequence_length=sequence_length, target_length=target_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    for _, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        pred, pred_hat = model(x)
    return pred, y

def test_lorenz(model, data_length,batch_size, sequence_length,target_length,shuffle):
    
    # Load and preprocess the Rossler Attractor dataset
    test_data = load_lorenz_attractor_dataset(data_length)

    # Create a DataLoader for the dataset
    test_dataset = LorenzAttractorDataset(test_data, sequence_length=sequence_length, target_length=target_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    for _, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        pred, pred_hat = model(x)
    return pred, y

def train_figure8_rnn(
    hidden_dim=64,
    data_length=5000,
    batch_size=5000,
    sequence_length=4,
    target_length=1,
    num_epochs=100,
    lr=1e-3,
    noise_stddev=1e-4,
    lambda_reg=1e-8,
    device="cuda",
    shuffle=False,
    s_init=0.1,
    beta_s=0.99,
    beta_lr=0.999,
    results_dir="results/rnn/wp/",
    tensorboard_name="fig8_wp"
):
    os.makedirs(results_dir, exist_ok=True)
    # Hyperparameters - figure8
    input_size = 2
    output_size = 2

    # Load and preprocess the Figure-8 dataset
    data = load_figure8_dataset(data_length)

    # Create a DataLoader for the dataset
    dataset = Figure8Dataset(data, sequence_length, target_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize the model inside the WeightPerturbation class
    model = RNNModel(input_size, output_size, hidden_dim).to(device)
    model.requires_grad_(False).eval()
    # Optimizer
    optimizer = Dopamine(lr=lr, s_init=s_init, beta_s=beta_s, beta_lr=beta_lr)

    # Initialize the WeightPerturbation class
    wp = WeightPerturbation(
        model,
        optimizer,
        noise_mean=0.0,
        noise_stddev=noise_stddev,
        lr=lr,
        lambda_reg=lambda_reg,
    )

    criterion = nn.MSELoss()

    writer = SummaryWriter("runs/"+tensorboard_name)

    # Lists to store training losses and rewards
    train_losses = []
    train_rewards = []

    # Training loop
    for epoch in range(num_epochs):
        # Lists to store losses and rewards for each batch
        losses = []
        rewards = []

        # Iterate over each batch
        for batch, (x, y) in enumerate(dataloader):
            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            pred, pred_hat = wp(x)

            # Calculate loss
            loss = criterion(pred, y.squeeze())
            loss_hat = criterion(pred_hat, y.squeeze())

            # Calculate reward
            reward = loss_hat - loss

            # Backward pass
            wp.backward(reward)

            losses.append(loss.item())
            rewards.append(reward.item())

        # Calculate average loss and reward for the epoch
        train_loss = np.mean(losses)
        train_reward = np.mean(rewards)

        # Print epoch and loss
        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Reward: {train_reward:.4f}"
        )

        # Append loss and reward to the lists
        train_losses.append(train_loss)
        train_rewards.append(train_reward)

    # Save the model
    torch.save(wp.model.state_dict(), results_dir + "fig8_rnn_wp.pth")


def train_rossler_rnn(
    hidden_dim=64,
    data_length=50000,
    batch_size=50000,
    sequence_length=1,
    target_length=1,
    num_epochs=500,
    lr=1e-3,
    noise_stddev=1e-4,
    lambda_reg=1e-8,
    device=device,
    shuffle=False,
    s_init=0.01,
    sr=1.0,
    beta_s=0.9,
    beta_lr=0.999,
    results_dir="results/rnn/wp/",
    tensorboard_name = "rossler_rnn_wp_relu_sq_1_bs_50000_lr_1e3_s_0.01"
):
    os.makedirs(results_dir, exist_ok=True)

    # Hyperparameters - rossler
    input_size = 3
    output_size = 3

    # Load and preprocess the Rossler Attractor dataset
    data = load_rossler_attractor_dataset(data_length)

    # Create a DataLoader for the dataset
    dataset = RosslerAttractorDataset(data, sequence_length, target_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize the model
    model = torch.nn.DataParallel(RNNModel(input_size, output_size, hidden_dim)).to(device)
    model.requires_grad_(False).eval()

    # Optimizer
    optimizer = Dopamine(lr=lr, s_init=s_init, beta_s=beta_s, beta_lr=beta_lr)

    # Initialize the WeightPerturbation class
    wp = WeightPerturbation(
        model,
        optimizer,
        noise_mean=0.0,
        noise_stddev=noise_stddev,
        lr=lr,
        sr=sr,
        lambda_reg=lambda_reg,
    )

    criterion = nn.MSELoss()
    writer = SummaryWriter("runs/"+tensorboard_name)

    # Lists to store training losses and rewards
    train_losses = []
    train_rewards = []

    # Training loop
    for epoch in range(num_epochs):
        # Lists to store losses and rewards for each batch
        losses = []
        rewards = []

        # Iterate over each batch
        for _, (x, y) in enumerate(dataloader):
            # Move data to device
            x = x.to(device)
            y = y.squeeze().to(device)

            # Forward pass
            pred, pred_hat = wp(x)

            # Calculate loss
            loss = criterion(pred, y)
            loss_hat = criterion(pred_hat, y)

            # Calculate reward
            reward = loss_hat - loss

            # Backward pass
            wp.backward(reward)

            losses.append(loss.item())
            rewards.append(reward.item())

        # Calculate average loss and reward for the epoch
        train_loss = np.mean(losses)
        train_reward = np.mean(rewards)

        writer.add_scalar('Loss', loss.item() , epoch+1)
        writer.add_scalar('Reward', reward.item(), epoch+1)
        
        if (epoch+1)%50:

            # Load the model and generate predictions
            pred, y = test_rossler(model=wp, data_length=data_length,batch_size=data_length, sequence_length=sequence_length,target_length=1,shuffle=False)

            # Predictions in 3D
            fig1 = plt.figure()
            ax = fig1.add_subplot(111, projection='3d')
            pred = pred.cpu().numpy()  # Convert to NumPy array
            y = y.squeeze().cpu().numpy()
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker='.', s=1)
            writer.add_figure("Prediction", fig1, global_step=epoch+1)

            # Predictions in 2D
            # Create a time array for the x-axis
            time = range(len(pred))
            # Plot each feature separately in 2D with time on the x-axis
            fig2 = plt.figure(figsize=(15, 5))
    
            for idx,feature in enumerate(["X","Y","Z"]):
                plt.subplot(1, 3, idx + 1)
                plt.scatter(time[:5000], pred[:, idx][:5000], label="Predicted", marker='.', s=5)
                plt.scatter(time[:5000], y[:, idx][:5000], label = "Target", marker='.',s=5)
                plt.xlabel("Time")
                plt.ylabel(f"{feature}")
                plt.title(f"{feature} vs. Time")
                plt.legend()
            plt.tight_layout()
            writer.add_figure("Prediction 1D", fig2, global_step=epoch+1)

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Reward: {train_reward:.4f}"
        )

    # Save the model
    torch.save(wp.model.state_dict(), results_dir + "rossler_rnn_wp.pth")


# Function to train the model on Lorenz Attractor dataset
def train_lorenz_rnn(
    hidden_dim=64,
    data_length=50000,
    batch_size=50000,
    sequence_length=1,
    target_length=1,
    num_epochs=500,
    lr=1e-3,
    noise_stddev=1e-4,
    lambda_reg=1e-8,
    device=device,
    shuffle=False,
    s_init=0.01,
    sr=1.0,
    beta_s=0.9,
    beta_lr=0.999,
    results_dir=None, 
    tensorboard_name = None
):
    os.makedirs(results_dir, exist_ok=True)

    # Hyperparameters - lorenz
    input_size = 3
    output_size = 3

    # Load and preprocess the Rossler Attractor dataset
    data = load_lorenz_attractor_dataset(data_length)

    # Create a DataLoader for the dataset
    dataset = LorenzAttractorDataset(data, sequence_length, target_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize the model
    model = torch.nn.DataParallel(RNNModel(input_size, output_size, hidden_dim)).to(device)
    model.requires_grad_(False).eval()

    # Optimizer
    optimizer = Dopamine(lr=lr, s_init=s_init, beta_s=beta_s, beta_lr=beta_lr)

    # Initialize the WeightPerturbation class
    wp = WeightPerturbation(
        model,
        optimizer,
        noise_mean=0.0,
        noise_stddev=noise_stddev,
        lr=lr,
        sr=sr,
        lambda_reg=lambda_reg,
    )

    criterion = nn.MSELoss()
    writer = SummaryWriter("runs/"+tensorboard_name)

    # Lists to store training losses and rewards
    train_losses = []
    train_rewards = []

    # Training loop
    for epoch in range(num_epochs):

        losses = []
        rewards = []
        # Iterate over each batch
        for _, (x, y) in enumerate(dataloader):
            # Move data to device
            x = x.to(device)
            y = y.squeeze().to(device)

            # Forward pass
            pred, pred_hat = wp(x)

            # Calculate loss
            loss = criterion(pred, y)
            loss_hat = criterion(pred_hat, y)

            # Calculate reward
            reward = loss_hat - loss

            # Backward pass
            wp.backward(reward)

            losses.append(loss.item())
            rewards.append(reward.item())

        # Calculate average loss and reward for the epoch
        train_loss = np.mean(losses)
        train_reward = np.mean(rewards)

        writer.add_scalar('Loss', loss.item() , epoch+1)
        writer.add_scalar('Reward', reward.item(), epoch+1)
        train_losses.append(train_loss)
        if (epoch+1)%50:

            # Load the model and generate predictions
            pred, y = test_lorenz(model=wp, data_length=data_length,batch_size=data_length, sequence_length=sequence_length,target_length=1,shuffle=False)

            # Plot Predictions in 3D
            fig1 = plt.figure()
            ax = fig1.add_subplot(111, projection='3d')
            pred = pred.cpu().numpy()  # Convert to NumPy array
            y = y.squeeze().cpu().numpy()
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker='.', s=1)
            writer.add_figure("Prediction", fig1, global_step=epoch+1)

            
            # Plot Predictions in 2D
            # Create a time array for the x-axis
            time = range(len(pred))
            # Plot each feature separately in 2D with time on the x-axis
            fig2 = plt.figure(figsize=(15, 5))
    
            for idx,feature in enumerate(["X","Y","Z"]):
                plt.subplot(1, 3, idx + 1)
                plt.scatter(time[:5000], pred[:, idx][:5000], label="Predicted", marker='.', s=5)
                plt.scatter(time[:5000], y[:, idx][:5000], label = "Target", marker='.',s=5)
                plt.xlabel("Time")
                plt.ylabel(f"{feature}")
                plt.legend()
            plt.tight_layout()
            writer.add_figure("Prediction 1D", fig2, global_step=epoch+1)

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Reward: {train_reward:.4f}"
        )

    # Save the model
    torch.save(wp.model.state_dict(), results_dir + "lorenz_rnn_wp.pth")
    np.save(results_dir + "lorenz_wp.npy",train_losses)
    # Test the model and save the predictions as figure
    
    # Load the model and generate predictions
    pred, y = test_lorenz(model=wp, data_length=data_length,batch_size=data_length, sequence_length=sequence_length,target_length=1,shuffle=False)

    # Plot Predictions in 3D
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    pred = pred.cpu().numpy()  # Convert to NumPy array
    y = y.squeeze().cpu().numpy()
    
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker='.', s=10, label="Prediction")
    ax.scatter(y[:, 0], y[:, 1], y[:, 2], marker='.', s=10, label="Target")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    color_tuple = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(color_tuple)
    ax.yaxis.set_pane_color(color_tuple)
    ax.zaxis.set_pane_color(color_tuple)
    ax.xaxis.line.set_color(color_tuple)
    ax.yaxis.line.set_color(color_tuple)
    ax.zaxis.line.set_color(color_tuple)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.legend(frameon=False)
    plt.title("WP")
    plt.tight_layout()
    # Save the figure
    fig1.savefig(results_dir + "lorenz_rnn_wp_3d.png")
    
    # Plot Predictions in 2D
    # Create a time array for the x-axis
    time = range(len(pred))
    # Plot each feature separately in 2D with time on the x-axis
    fig2 = plt.figure(figsize=(15, 5))
    for idx,feature in enumerate(["X","Y","Z"]):
        plt.subplot(1, 3, idx + 1)
        plt.scatter(time[:5000], pred[:, idx][:5000], label="Predicted", marker='.', s=5)
        plt.scatter(time[:5000], y[:, idx][:5000], label = "Target", marker='.',s=5)
        plt.xlabel("Time")
        plt.ylabel(f"{feature}")
        plt.legend()
    plt.tight_layout()
    
    # Save the figure
    fig2.savefig(results_dir + "lorenz_rnn_wp_1d.png")
    
if __name__ == "__main__":

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    torch.set_num_threads(4)

    torch.cuda.empty_cache() 
    
    train_lorenz_rnn(
            hidden_dim=512,
            data_length=5000,
            batch_size=5000,
            sequence_length=32,
            target_length=1,
            num_epochs=2000,
            lr=1e-3,
            noise_stddev=3e-4,
            lambda_reg=1e-8,
            device=device,
            shuffle=False,
            s_init=25e-3,
            sr = 1.1,
            beta_s=0.99,
            beta_lr=0.4,
            results_dir="results/rnn/wp_new/lorenz_WP",
            tensorboard_name = "lorenz_WP",
        )
    print("**** End of training Lorenz ****")

                
