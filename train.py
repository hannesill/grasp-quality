import argparse
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np

from dataset import OptimizedGraspDataset
from model import GQEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="Train Grasp Quality Estimator")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_size', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--base_channels', type=int, default=4, help='Base channels for the CNN')
    parser.add_argument('--fc_dims', nargs='+', type=int, default=[32, 16], help='Dimensions of FC layers in the head')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--wandb_entity', type=str, default='tairo', help='WandB entity')
    parser.add_argument('--project_name', type=str, default='adlr', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.01):
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
    dataset = OptimizedGraspDataset(data_path)

    num_samples = len(dataset)
    print(f"Total samples: {num_samples}")

    # Create indices and shuffle
    indices = list(range(num_samples))
    random.seed(42)
    random.shuffle(indices)

    # Split into train and validation
    train_indices = indices[:args.train_size]
    val_indices = indices[-args.val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(f"Train dataset size: {len(train_set)}, Validation dataset size: {len(val_set)}")

    # Create simple DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=True,
        prefetch_factor=2
    )

    # --- Model, Optimizer, Loss ---
    model = GQEstimator(input_size=48, base_channels=args.base_channels, fc_dims=args.fc_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    wandb.watch(model, criterion, log="all", log_freq=100)

    # --- Training Loop ---
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # === TRAINING ===
        model.train()
        total_train_loss = 0
        num_train_samples = 0
        
        # Timing variables
        data_loading_time = 0
        sdf_loading_time = 0
        sdf_encoding_time = 0
        forward_pass_time = 0
        backward_pass_time = 0
        
        training_start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # === DATA LOADING ===
            data_start = time.time()
            scene_indices = batch['scene_idx'].to(device)
            grasp_batch = batch['grasp'].to(device)
            score_batch = batch['score'].to(device)
            data_loading_time += time.time() - data_start
            
            # === SDF LOADING ===
            sdf_load_start = time.time()
            # Load SDFs from disk (naive approach)
            sdf_list = []
            for scene_idx in scene_indices:
                sdf = dataset.get_sdf(scene_idx.item())
                sdf_list.append(sdf)
            sdf_batch = torch.stack(sdf_list).to(device)
            sdf_loading_time += time.time() - sdf_load_start
            
            # === SDF ENCODING ===
            sdf_encode_start = time.time()
            sdf_features_batch = model.encode_sdf(sdf_batch)
            sdf_encoding_time += time.time() - sdf_encode_start
            
            # === FORWARD PASS ===
            forward_start = time.time()
            flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
            pred_quality = model(flattened_features)
            loss = criterion(pred_quality, score_batch)
            forward_pass_time += time.time() - forward_start
            
            # === BACKWARD PASS ===
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            backward_pass_time += time.time() - backward_start
            
            # Update metrics
            total_train_loss += loss.item() * sdf_features_batch.size(0)
            num_train_samples += sdf_features_batch.size(0)
            
            running_loss = total_train_loss / num_train_samples
            batch_time = time.time() - batch_start_time
            
            pbar.set_postfix(loss=f'{running_loss:.4f}', batch_time=f'{batch_time:.3f}s')

        avg_train_loss = total_train_loss / num_train_samples
        training_time = time.time() - training_start_time
        
        # === VALIDATION ===
        model.eval()
        total_val_loss = 0
        num_val_samples = 0
        
        validation_start_time = time.time()
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for batch in pbar_val:
                # Simple validation forward pass
                scene_indices = batch['scene_idx'].to(device)
                grasp_batch = batch['grasp'].to(device)
                score_batch = batch['score'].to(device)
                
                # Load SDFs
                sdf_list = []
                for scene_idx in scene_indices:
                    sdf = dataset.get_sdf(scene_idx.item())
                    sdf_list.append(sdf)
                sdf_batch = torch.stack(sdf_list).to(device)
                
                # Forward pass
                sdf_features_batch = model.encode_sdf(sdf_batch)
                flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
                pred_quality = model(flattened_features)
                loss = criterion(pred_quality, score_batch)

                total_val_loss += loss.item() * sdf_features_batch.size(0)
                num_val_samples += sdf_features_batch.size(0)
                
                running_val_loss = total_val_loss / num_val_samples
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_val_samples
        validation_time = time.time() - validation_start_time
        
        # Total epoch time
        epoch_time = time.time() - epoch_start_time
        
        # === TIMING BREAKDOWN ===
        print(f"\n=== EPOCH {epoch+1} TIMING BREAKDOWN ===")
        print(f"Total epoch time: {epoch_time:.2f}s")
        print(f"Training time: {training_time:.2f}s ({training_time/epoch_time*100:.1f}%)")
        print(f"Validation time: {validation_time:.2f}s ({validation_time/epoch_time*100:.1f}%)")
        
        print(f"\nTraining timing breakdown:")
        print(f"  Data loading: {data_loading_time:.2f}s ({data_loading_time/training_time*100:.1f}%)")
        print(f"  SDF loading: {sdf_loading_time:.2f}s ({sdf_loading_time/training_time*100:.1f}%)")
        print(f"  SDF encoding: {sdf_encoding_time:.2f}s ({sdf_encoding_time/training_time*100:.1f}%)")
        print(f"  Forward pass: {forward_pass_time:.2f}s ({forward_pass_time/training_time*100:.1f}%)")
        print(f"  Backward pass: {backward_pass_time:.2f}s ({backward_pass_time/training_time*100:.1f}%)")
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{args.epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
            "timing/epoch_total": epoch_time,
            "timing/training_total": training_time,
            "timing/validation_total": validation_time,
            "timing/data_loading": data_loading_time,
            "timing/sdf_loading": sdf_loading_time,
            "timing/sdf_encoding": sdf_encoding_time,
            "timing/forward_pass": forward_pass_time,
            "timing/backward_pass": backward_pass_time,
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