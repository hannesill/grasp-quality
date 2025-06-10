import random
from pathlib import Path
from dataset import GraspDataset
from model import GQEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch
import wandb

# --- Configuration ---
LEARNING_RATE = 1e-4
EPOCHS = 100
VAL_SPLIT = 0.2
BASE_CHANNELS = 16
FC_DIMS = [256, 128, 64]

# Initialize wandb
wandb.init(
    project="grasp-quality-estimator",
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "val_split": VAL_SPLIT,
        "architecture": "GQEstimator",
        "dataset": "GraspNet-1Billion-Subset",
        "base_channels": BASE_CHANNELS,
        "fc_dims": FC_DIMS
    }
)

# Load the dataset
data_path = Path('data/processed')
dataset = GraspDataset(data_path)

# Split dataset into training and validation
num_samples = len(dataset)
train_size = int(num_samples * (1 - VAL_SPLIT))
val_size = num_samples - train_size

print(f"Total scenes: {num_samples}, Train scenes: {train_size}, Val scenes: {val_size}")

# Shuffle indices
random.seed(42)
indices = list(range(num_samples))
random.shuffle(indices)

# Create Subsets using the full dataset indices
train_dataset = torch.utils.data.Subset(dataset, indices[:100])
val_dataset = torch.utils.data.Subset(dataset, indices[-10:])

print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

# Initialize model and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GQEstimator(
    input_size=48,
    base_channels=BASE_CHANNELS,
    fc_dims=FC_DIMS
).to(device)

# Log model architecture and gradients to wandb
wandb.watch(model, log="all", log_freq=100)

# Training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

print(f"\nStarting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    total_train_grasps = 0

    # Iterate over scenes for training
    for scene in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False):
        num_grasps_in_scene = len(scene['grasps'])
        if num_grasps_in_scene == 0:
            continue
        
        total_train_grasps += num_grasps_in_scene
        sdf = scene['sdf'].float().to(device)

        for i in range(num_grasps_in_scene):
            hand_pose = scene['grasps'][i].float().to(device)
            scores = scene['scores'][i].float().to(device).unsqueeze(0)

            optimizer.zero_grad()
            pred_quality = model(sdf, hand_pose)
            loss = criterion(pred_quality, scores)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

    avg_epoch_train_loss = epoch_train_loss / total_train_grasps if total_train_grasps > 0 else 0

    # Validation loop
    model.eval()
    epoch_val_loss = 0.0
    total_val_grasps = 0
    with torch.no_grad():
        for scene in tqdm(val_dataset, desc=f"Epoch {epoch+1}/{EPOCHS} Validation", leave=False):
            num_grasps_in_scene = len(scene['grasps'])
            if num_grasps_in_scene == 0:
                continue

            total_val_grasps += num_grasps_in_scene
            sdf = scene['sdf'].float().to(device)

            for i in range(num_grasps_in_scene):
                hand_pose = scene['grasps'][i].float().to(device)
                scores = scene['scores'][i].float().to(device).unsqueeze(0)
                
                pred_quality = model(sdf, hand_pose)
                loss = criterion(pred_quality, scores)
                epoch_val_loss += loss.item()

    avg_epoch_val_loss = epoch_val_loss / total_val_grasps if total_val_grasps > 0 else 0

    print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}')

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_epoch_train_loss,
        "val_loss": avg_epoch_val_loss
    })

# Save model
model_path = "model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully to {model_path}")

# Log model artifact to wandb
artifact = wandb.Artifact('gq-estimator', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()