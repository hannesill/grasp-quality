import argparse
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np

from dataset import ThreeChannelGPUDataset, ThreeChannelPreprocessedDataset
from model import GQEstimator, get_model_info

def parse_args():
    parser = argparse.ArgumentParser(description="Train Optimized 3-Channel Grasp Quality Estimator")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_size', type=int, default=100000, help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=10000, help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--preprocessed_path', type=str, default='data/preprocessed_3channel', 
                       help='Path to preprocessed 3-channel data')
    parser.add_argument('--wandb_entity', type=str, default='tairo', help='WandB entity')
    parser.add_argument('--project_name', type=str, default='adlr', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--use_gpu_cache', action='store_true', help='Load all data to GPU memory')
    parser.add_argument('--cache_size', type=int, default=100, help='File cache size for on-demand loading')
    
    # Model architecture arguments (with optimized defaults)
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for the CNN')
    parser.add_argument('--fc_dims', nargs='+', type=int, default=[512, 256, 128], 
                       help='Dimensions of FC layers')
    
    return parser.parse_args()

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
    print("\n🚀 Creating optimized 3-channel model...")
    model = GQEstimator(
        input_size=48,
        base_channels=args.base_channels,
        fc_dims=args.fc_dims
    ).to(device)
    
    # Show model info
    info = get_model_info()
    print(f"\n📊 Model Architecture:")
    print(f"  • {info['architecture']}")
    print(f"  • Input: {info['input_channels']} channels (SDF + palm dist + fingertip dist)")
    print(f"  • Channel progression: {info['channel_progression']}")
    print(f"  • Final spatial resolution: {info['spatial_resolution']}")
    print(f"  • Features: {', '.join(info['features'])}")
    
    # --- Dataset Creation ---
    print(f"\n🚀 Creating 3-channel dataset from {args.preprocessed_path}...")
    dataset_start = time.time()
    
    if args.use_gpu_cache:
        print("📦 Using GPU-cached dataset (all data in GPU memory)")
        dataset = ThreeChannelGPUDataset(Path(args.preprocessed_path), device=device)
    else:
        print("📦 Using on-demand dataset with file caching")
        dataset = ThreeChannelPreprocessedDataset(Path(args.preprocessed_path), cache_size=args.cache_size)
    
    dataset_time = time.time() - dataset_start
    print(f"Dataset creation time: {dataset_time:.2f}s")
    
    # --- Train/Val Splits ---
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.seed(42)  # For reproducible splits
    random.shuffle(indices)
    
    # Limit dataset size if requested
    if args.train_size + args.val_size > num_samples:
        print(f"⚠️  Requested {args.train_size + args.val_size} samples but only {num_samples} available")
        args.train_size = min(args.train_size, num_samples - args.val_size)
        args.val_size = min(args.val_size, num_samples - args.train_size)
    
    train_set = Subset(dataset, indices[:args.train_size])
    val_set = Subset(dataset, indices[-args.val_size:])
    print(f"Train dataset size: {len(train_set)}, Validation dataset size: {len(val_set)}")
    
    # --- DataLoaders ---
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False
    )
    
    # --- Training Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
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
        config={**vars(args), **info}
    )
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # --- Training Loop ---
    print(f"\n🏃‍♂️ Starting training for {args.epochs} epochs...")
    print("⚡ Optimized 3-channel architecture for spatial grasp understanding")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # === TRAINING PHASE ===
        model.train()
        total_train_loss = 0
        num_train_samples = 0
        
        # Timing variables
        data_loading_time = 0
        forward_pass_time = 0
        backward_pass_time = 0
        
        training_start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            # === DATA LOADING ===
            data_start = time.time()
            input_batch = batch['input']  # (B, 3, 48, 48, 48)
            score_batch = batch['score']  # (B,)
            
            # Move to device if not already there
            if input_batch.device != device:
                input_batch = input_batch.to(device)
            if score_batch.device != device:
                score_batch = score_batch.to(device)
            
            data_loading_time += time.time() - data_start
            
            # === FORWARD PASS ===
            forward_start = time.time()
            pred_quality = model.forward_with_3channel_input(input_batch)
            loss = criterion(pred_quality, score_batch)
            forward_pass_time += time.time() - forward_start
            
            # === BACKWARD PASS ===
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_pass_time += time.time() - backward_start
            
            # Update metrics
            batch_size = input_batch.size(0)
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
                # Validation forward pass
                input_batch = batch['input']
                score_batch = batch['score']
                
                # Move to device if not already there
                if input_batch.device != device:
                    input_batch = input_batch.to(device)
                if score_batch.device != device:
                    score_batch = score_batch.to(device)
                
                pred_quality = model.forward_with_3channel_input(input_batch)
                loss = criterion(pred_quality, score_batch)

                batch_size = input_batch.size(0)
                total_val_loss += loss.item() * batch_size
                num_val_samples += batch_size
                
                running_val_loss = total_val_loss / num_val_samples
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_val_samples
        validation_time = time.time() - validation_start_time
        
        # Total epoch time
        epoch_time = time.time() - epoch_start_time
        
        # === PERFORMANCE ANALYSIS ===
        print(f"\n⚡ OPTIMIZED 3-CHANNEL TRAINING BREAKDOWN")
        print(f"Total epoch time: {epoch_time:.2f}s")
        print(f"Training time: {training_time:.2f}s ({training_time/epoch_time*100:.1f}%)")
        print(f"Validation time: {validation_time:.2f}s ({validation_time/epoch_time*100:.1f}%)")
        
        print(f"\nTraining phase breakdown:")
        print(f"  Data loading: {data_loading_time:.4f}s ({data_loading_time/training_time*100:.2f}%)")
        print(f"  Forward pass: {forward_pass_time:.2f}s ({forward_pass_time/training_time*100:.1f}%)")
        print(f"  Backward pass: {backward_pass_time:.2f}s ({backward_pass_time/training_time*100:.1f}%)")
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{args.epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_optimized.pth')
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
        })
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    # --- Save Final Model ---
    model_path = f"{wandb.run.dir}/final_model_optimized.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model saved to best_model_optimized.pth")
    
    # --- Final Performance Summary ---
    print(f"\n🎉 OPTIMIZED 3-CHANNEL TRAINING COMPLETE! 🎉")
    print(f"🔥 Architecture: {info['architecture']}")
    print(f"📊 Training samples: {args.train_size}")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.use_gpu_cache:
        print(f"📱 GPU Memory Usage: {dataset._get_gpu_memory_usage():.2f} GB")
    
    print(f"⚡ Model optimized for computation vs. results balance!")
    print(f"🚀 Ready for online grasp optimization!")
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args) 