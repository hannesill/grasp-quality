import argparse
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np

from dataset import GPUCachedGraspDataset
from model import GQEstimator

"""
This script trains a Grasp Quality Estimator (GQEstimator) using a GPU-cached dataset.
It supports loading pre-trained model weights and logging to Weights & Biases (WandB).

Usage Examples:
  # Start new training:
  python train.py --epochs 100 --batch_size 128 --lr 1e-3
  
  # Load pre-trained weights and continue training:
  python train.py --resume_from_saved_model best_model.pth --epochs 100
  
  # Train with custom model size:
  python train.py --base_channels 16 --fc_dims 512 256 128 --batch_size 64
  
  # Train with large dataset:
  python train.py --train_size 1000000 --val_size 100000 --batch_size 256
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train Grasp Quality Estimator")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_size', type=int, default=5000000, help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=500000, help='Number of validation samples')
    parser.add_argument('--base_channels', type=int, default=8, help='Base channels for the CNN')
    parser.add_argument('--fc_dims', nargs='+', type=int, default=[256, 128, 64], help='Dimensions of FC layers')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers (0 recommended for GPU cached data)')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--wandb_entity', type=str, default='tairo', help='WandB entity')
    parser.add_argument('--project_name', type=str, default='adlr', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--resume_from_saved_model', type=str, default=None, help='Path to saved model weights (.pth file) to initialize the model with')
    return parser.parse_args()

def load_model_weights(model_path, model, device):
    """Load model weights from a saved .pth file."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"🔄 Loading model weights from {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # Full checkpoint format
            state_dict = state_dict['model_state_dict']
        
        # Handle compiled model prefix (_orig_mod.)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("   Detected compiled model weights, removing _orig_mod. prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load the cleaned state dict
        model.load_state_dict(state_dict)
            
        print(f"✅ Model weights loaded successfully!")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}. "
                         f"This might be due to model architecture mismatch.")

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
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
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU for optimal performance.")
        return

    # --- Model Creation ---
    model = GQEstimator(
        input_size=48, 
        base_channels=args.base_channels, 
        fc_dims=args.fc_dims
    ).to(device)
    
    # Load pre-trained weights if specified
    if args.resume_from_saved_model:
        load_model_weights(args.resume_from_saved_model, model, device)
    
    # --- Dataset Creation with GPU Caching ---
    print("\n🚀 Creating GPU-cached datasets (train/val) using splits.json...")
    ds_create_start = time.time()
    train_dataset = GPUCachedGraspDataset(Path(args.data_path), device=device, split='train', splits_file='data/splits.json')
    train_create_time = time.time() - ds_create_start
    print(f"Train split loaded in: {train_create_time:.2f}s")

    ds_val_start = time.time()
    val_dataset = GPUCachedGraspDataset(Path(args.data_path), device=device, split='val', splits_file='data/splits.json')
    val_create_time = time.time() - ds_val_start
    print(f"Val split loaded in: {val_create_time:.2f}s")

    # --- Optional size caps ---
    random.seed(42)
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    val_indices = list(range(len(val_dataset)))
    random.shuffle(val_indices)

    if args.train_size is not None:
        train_indices = train_indices[:args.train_size]
    if args.val_size is not None:
        val_indices = val_indices[:args.val_size]

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    print(f"Train dataset size: {len(train_set)}, Validation dataset size: {len(val_set)}")
    
    # --- DataLoaders ---
    # Note: num_workers=0 is recommended since data is already on GPU
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False  # Data is already on GPU
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False  # Data is already on GPU
    )
    
    # --- Training Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=args.epochs)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    # --- Model Optimization ---
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✅ Model compiled with PyTorch 2.0 for additional speedup")
        except Exception as e:
            print(f"⚠️  Could not compile model: {e}")
    
    # --- WandB Initialization ---
    wandb.init(
        entity=args.wandb_entity,
        project=args.project_name,
        name=args.run_name,
        config=args
    )
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # --- Training Loop ---
    if args.resume_from_saved_model:
        print(f"Starting training for {args.epochs} epochs with pre-trained weights...")
    else:
        print(f"Starting training for {args.epochs} epochs from scratch...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # === TRAINING PHASE ===
        model.train()
        total_train_loss = 0
        num_train_samples = 0
        
        # Timing variables for detailed analysis
        data_loading_time = 0
        forward_pass_time = 0
        backward_pass_time = 0
        
        training_start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            # === INSTANT DATA ACCESS (already on GPU!) ===
            data_start = time.time()
            sdf_batch = batch['sdf']
            grasp_batch = batch['grasp']
            score_batch = batch['score']
            data_loading_time += time.time() - data_start
            
            # === FORWARD PASS ===
            forward_start = time.time()
            pred_quality = model.forward_with_sdf(sdf_batch, grasp_batch)
            loss = criterion(pred_quality, score_batch)
            forward_pass_time += time.time() - forward_start
            
            # === BACKWARD PASS ===
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_pass_time += time.time() - backward_start
            
            # Update metrics
            batch_size = sdf_batch.size(0)
            total_train_loss += loss.item() * batch_size
            num_train_samples += batch_size
            
            running_loss = total_train_loss / num_train_samples
            batch_time = time.time() - batch_start_time
            
            pbar.set_postfix(loss=f'{running_loss:.4f}', batch_time=f'{batch_time:.3f}s')

        avg_train_loss = total_train_loss / num_train_samples
        training_time = time.time() - training_start_time
        
        # === VALIDATION PHASE ===
        model.eval()
        total_val_loss = 0
        num_val_samples = 0
        
        validation_start_time = time.time()
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for batch in pbar_val:
                # Validation forward pass (also instant!)
                sdf_batch = batch['sdf']
                grasp_batch = batch['grasp']
                score_batch = batch['score']
                
                pred_quality = model.forward_with_sdf(sdf_batch, grasp_batch)
                loss = criterion(pred_quality, score_batch)

                batch_size = sdf_batch.size(0)
                total_val_loss += loss.item() * batch_size
                num_val_samples += batch_size
                
                running_val_loss = total_val_loss / num_val_samples
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_val_samples
        validation_time = time.time() - validation_start_time
        
        # Total epoch time
        epoch_time = time.time() - epoch_start_time
        
        # === PERFORMANCE ANALYSIS ===
        print(f"\nTRAINING BREAKDOWN ⚡")
        print(f"Total epoch time: {epoch_time:.2f}s")
        print(f"Training time: {training_time:.2f}s ({training_time/epoch_time*100:.1f}%)")
        print(f"Validation time: {validation_time:.2f}s ({validation_time/epoch_time*100:.1f}%)")
        
        print(f"\nTraining phase breakdown:")
        print(f"  Data loading: {data_loading_time:.4f}s ({data_loading_time/training_time*100:.2f}%)")
        print(f"  Forward pass: {forward_pass_time:.2f}s ({forward_pass_time/training_time*100:.1f}%)")
        print(f"  Backward pass: {backward_pass_time:.2f}s ({backward_pass_time/training_time*100:.1f}%)")
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{args.epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save to local directory (will be overwritten by other runs)
            torch.save(model.state_dict(), 'best_model.pth')
            # Also save to wandb run directory (unique per run)
            best_model_path = f"{wandb.run.dir}/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            wandb.save('best_model.pth')
            print(f"✅ New best model saved! Val Loss: {avg_val_loss:.4f}")
        
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
            "timing/forward_pass": forward_pass_time,
            "timing/backward_pass": backward_pass_time,
            "gpu_memory_usage_gb": (train_dataset._get_gpu_memory_usage() + val_dataset._get_gpu_memory_usage()),
        })
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    # --- Save Final Model ---
    model_path = f"{wandb.run.dir}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model saved to best_model.pth (local) and {wandb.run.dir}/best_model.pth (wandb)")
    
    print(f"\n🎉 TRAINING COMPLETE! 🎉")
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)