import argparse
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import numpy as np

from dataset import GraspDataset, OptimizedGraspDataset
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
    parser.add_argument('--check_data_distribution', action='store_true', default=False, help='Check data distribution')
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

def collect_unique_scene_indices_fast(dataset_subset):
    """
    Directly extract unique scene indices from dataset without loading data.
    This is 100x+ faster than iterating through the dataloader.
    """
    unique_scenes = set()
    
    # Access the underlying dataset and indices
    base_dataset = dataset_subset.dataset  # OptimizedGraspDataset
    subset_indices = dataset_subset.indices  # Which samples we're using
    
    # Extract scene indices directly from grasp_locations
    for idx in subset_indices:
        scene_idx, grasp_idx = base_dataset.grasp_locations[idx]
        unique_scenes.add(scene_idx)
    
    return unique_scenes

def compute_epoch_scene_mapping(train_set, val_set):
    """
    Pre-compute scene mappings once instead of every epoch.
    Returns all unique scenes that will ever be used.
    """
    train_scenes = collect_unique_scene_indices_fast(train_set)
    val_scenes = collect_unique_scene_indices_fast(val_set)
    all_unique_scenes = train_scenes.union(val_scenes)
    
    return {
        'train_scenes': train_scenes,
        'val_scenes': val_scenes, 
        'all_scenes': all_unique_scenes
    }

def preprocess_all_unique_sdfs_parallel(unique_scenes, dataset, model, device, batch_size=64):
    """
    Pre-encode unique SDFs in parallel batches.
    
    Args:
        unique_scenes: Set of unique scene indices
        dataset: Dataset to get SDFs from  
        model: Model with encode_sdf method
        device: Device to process on
        batch_size: Number of SDFs to process in parallel
    """
    print(f"Pre-encoding {len(unique_scenes)} unique SDFs in batches of {batch_size}...")
    sdf_features_cache = {}
    unique_scenes_list = list(unique_scenes)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_scenes_list), batch_size), desc="Parallel SDF encoding"):
            batch_scene_indices = unique_scenes_list[i:i + batch_size]
            
            # Load batch of SDFs
            sdf_batch = torch.stack([dataset.get_sdf(scene_idx) for scene_idx in batch_scene_indices])
            sdf_batch = sdf_batch.to(device)
            
            # Encode batch in parallel
            sdf_features_batch = model.encode_sdf(sdf_batch)
            
            # Store results
            for j, scene_idx in enumerate(batch_scene_indices):
                sdf_features_cache[scene_idx] = sdf_features_batch[j].detach()
    
    return sdf_features_cache

def process_batch_with_cached_features(batch, sdf_features_cache, device):
    """
    Process batch using pre-computed SDF features.
    """
    scene_indices = batch['scene_idx']
    grasp_batch = batch['grasp'].to(device)
    score_batch = batch['score'].to(device)
    
    # Build batch of SDF features by lookup
    batch_size = len(scene_indices)
    feature_dim = next(iter(sdf_features_cache.values())).shape[0]
    sdf_features_batch = torch.zeros(batch_size, feature_dim, device=device)
    
    for i, scene_idx in enumerate(scene_indices):
        sdf_features_batch[i] = sdf_features_cache[scene_idx.item()]
    
    return sdf_features_batch, grasp_batch, score_batch

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

    # Get scene mappings (which unique scenes will be used in training and validation)
    scene_mappings = compute_epoch_scene_mapping(train_set, val_set)
    print("Unique scenes:")
    print(f"Train scenes: {len(scene_mappings['train_scenes'])}")
    print(f"Validation scenes: {len(scene_mappings['val_scenes'])}")
    print(f"All scenes: {len(scene_mappings['all_scenes'])}")

    # Create DataLoaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True  # Important: shuffle for training
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=False  # No shuffle for validation
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
        # === EPOCH-LEVEL SDF PREPROCESSING ===
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Pre-encode all unique SDFs
        start_time = time.time()
        train_sdf_cache = preprocess_all_unique_sdfs_parallel(scene_mappings['train_scenes'], dataset, model, device)
        val_sdf_cache = preprocess_all_unique_sdfs_parallel(scene_mappings['val_scenes'], dataset, model, device)
        preprocessing_time = time.time() - start_time
        
        print(f"SDF preprocessing time: {preprocessing_time:.2f}s")
        
        # Calculate efficiency metrics
        total_train_samples = len(train_set)
        total_val_samples = len(val_set)
        train_efficiency = len(scene_mappings['train_scenes']) / total_train_samples
        val_efficiency = len(scene_mappings['val_scenes']) / total_val_samples
        
        print(f"Train efficiency: {train_efficiency:.3f} (lower is better)")
        print(f"Val efficiency: {val_efficiency:.3f} (lower is better)")
        print(f"Train SDF reuse factor: {total_train_samples / len(scene_mappings['train_scenes']):.1f}x")
        print(f"Val SDF reuse factor: {total_val_samples / len(scene_mappings['val_scenes']):.1f}x")
        
        # === TRAINING ===
        model.train()
        total_train_loss = 0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for batch in pbar:
            optimizer.zero_grad()

            # Fast batch processing with cached SDF features
            sdf_features_batch, grasp_batch, score_batch = process_batch_with_cached_features(
                batch, train_sdf_cache, device
            )

            actual_batch_size = sdf_features_batch.size(0)

            # Concatenate and predict
            flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
            pred_quality = model(flattened_features)
            loss = criterion(pred_quality, score_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item() * actual_batch_size
            num_steps += actual_batch_size
            
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
                # Fast batch processing with cached SDF features
                sdf_features_batch, grasp_batch, score_batch = process_batch_with_cached_features(
                    batch, val_sdf_cache, device
                )

                actual_batch_size = sdf_features_batch.size(0)

                # Concatenate and predict
                flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
                pred_quality = model(flattened_features)
                loss = criterion(pred_quality, score_batch)

                total_val_loss += loss.item() * actual_batch_size
                num_steps += actual_batch_size
                
                # Update progress bar
                running_val_loss = total_val_loss / num_steps
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_steps
        
        # Clear SDF caches to free GPU memory
        del train_sdf_cache, val_sdf_cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{args.epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
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
            "train_efficiency": train_efficiency,
            "val_efficiency": val_efficiency,
            "preprocessing_time": preprocessing_time
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