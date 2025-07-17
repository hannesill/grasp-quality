import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import GraspDataset
from src.model import GQEstimator

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Grasp Quality Estimator")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use. Defaults to all.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation data split ratio.')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for the CNN')
    parser.add_argument('--fc_dims', nargs='+', type=int, default=[32, 16], help='Dimensions of FC layers in the head')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--encoder_path', type=str, default=None, help='Path to pretrained encoder checkpoint')
    parser.add_argument('--wandb_entity', type=str, default='tairo', help='WandB entity')
    parser.add_argument('--project_name', type=str, default='adlr', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def main(args):
    # --- WandB Initialization ---
    wandb.init(
        entity=args.wandb_entity,
        project=args.project_name,
        name=args.run_name,
        config=args
    )

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    data_path = Path(args.data_path)
    dataset = GraspDataset(data_path)

    num_available = len(dataset)
    print(f"Total samples available: {num_available}")

    num_to_use = num_available if args.num_samples is None else min(args.num_samples, num_available)
    print(f"Using {num_to_use} samples.")

    # Create indices, shuffle and subsample
    indices = list(range(num_available))
    random.shuffle(indices)
    indices = indices[:num_to_use]

    # Split into train and validation
    val_size = int(num_to_use * args.val_split)
    train_size = num_to_use - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(f"Train dataset size: {len(train_set)}, Validation dataset size: {len(val_set)}")

    # Create DataLoaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True # Important: shuffle for training
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=False # No shuffle for validation
    )

    # --- Model, Optimizer, Loss ---
    model = GQEstimator(input_size=48, base_channels=args.base_channels, fc_dims=args.fc_dims).to(device)
    if args.encoder_path:
        model.object_encoder.load_state_dict(torch.load(args.encoder_path, weights_only=True))
        print(f"Loaded encoder from {args.encoder_path}")

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        print(f"Loaded model from {args.model_path}")
    elif not args.encoder_path:
        print("No model path provided, starting from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    
    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.7, patience=10
    # )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    wandb.watch(model, criterion, log="all", log_freq=100)

    # --- Training Loop ---
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # === EPOCH-LEVEL SDF PREPROCESSING ===
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # === TRAINING ===
        model.train()
        total_train_loss = 0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for batch in pbar:
            optimizer.zero_grad()

            # Move batch to device
            sdf_batch = batch['sdf'].to(device)
            grasp_batch = batch['grasp'].to(device)
            score_batch = batch['score'].to(device)

            # Encode SDFs and predict scores
            sdf_features_batch = model.encode_sdf(sdf_batch)
            flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
            pred_quality = model(flattened_features)

            loss = criterion(pred_quality, score_batch)
            loss.backward()
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()
            num_steps += 1
            
            # Update progress bar
            running_loss = total_train_loss / num_steps
            pbar.set_postfix(loss=f'{running_loss:.4f}')

        avg_train_loss = total_train_loss / num_steps

        # === VALIDATION ===
        model.eval()
        total_val_loss = 0
        num_steps = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for batch in pbar_val:
                # Move batch to device
                sdf_batch = batch['sdf'].to(device)
                grasp_batch = batch['grasp'].to(device)
                score_batch = batch['score'].to(device)

                # Encode SDFs and predict scores
                sdf_features_batch = model.encode_sdf(sdf_batch)
                flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
                pred_quality = model(flattened_features)

                loss = criterion(pred_quality, score_batch)

                total_val_loss += loss.item()
                num_steps += 1
                
                # Update progress bar
                running_val_loss = total_val_loss / num_steps
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_steps
        
        # Step the scheduler
        # scheduler.step(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{args.epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss and epoch > 10:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
        })
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- Save Final Model ---
    model_path = f"{wandb.run.dir}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model saved to best_model.pth")
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)