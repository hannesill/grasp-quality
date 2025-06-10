import random
from pathlib import Path
from dataset import GraspDataset
from model import GQEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch

# Load the dataset
data_path = Path('data/processed')
dataset = GraspDataset(data_path)

# Split dataset into training and validation
val_split = 0.2
num_samples = len(dataset)
train_size = int(num_samples * (1 - val_split))
val_size = num_samples - train_size

print(f"Subset samples: {num_samples}, Calculated train size: {train_size}, Calculated val size: {val_size}")

# Shuffle indices
random.seed(42)
indices = list(range(num_samples))
random.shuffle(indices)

# Split indices
train_indices = indices[:100]
val_indices = indices[-10:]

# Create Subsets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")


# Initialize model and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GQEstimator(
    input_size=48,
    base_channels=16,
    fc_dims=[256, 128, 64]
).to(device)

# Training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()
num_epochs = 100
train_losses = []
val_losses = []

print(f"\nStarting quick training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0

    # Iterate over scenes
    for scene in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
        num_samples = len(scene['grasps'])

        for i in range(num_samples):
            sdf = scene['sdf'].float().to(device)
            hand_pose = scene['grasps'][i].float().to(device)
            scores = scene['scores'][i].float().to(device).unsqueeze(0)

            optimizer.zero_grad()
            pred_quality = model(sdf, hand_pose)
            loss = criterion(pred_quality, scores)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

    avg_epoch_train_loss = epoch_train_loss / train_size
    train_losses.append(avg_epoch_train_loss)

    # Validation loop
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for scene in tqdm(val_dataset, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
            num_samples = len(scene['grasps'])

            for i in range(num_samples):
                sdf = scene['sdf'].float().to(device)
                hand_pose = scene['grasps'][i].float().to(device)
                scores = scene['scores'][i].float().to(device).unsqueeze(0)

            pred_quality = model(sdf, hand_pose)
            loss = criterion(pred_quality, scores)
            epoch_val_loss += loss.item()

    avg_epoch_val_loss = epoch_val_loss / val_size
    val_losses.append(avg_epoch_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}')

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()