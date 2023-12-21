import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from dopamine.datasets.rossler import RosslerAttractorDataset, load_rossler_attractor_dataset
from dopamine.models.rnn import RNNModel
from dopamine.optimizer import Dopamine
from dopamine.wp import WeightPerturbation
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training in {device}")

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