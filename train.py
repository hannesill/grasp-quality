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
    parser.add_argument('--sdf_batch_size', type=int, default=64, help='Batch size for SDF encoding')
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.01, min_delta_pct=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.min_delta_pct = min_delta_pct  # Percentage improvement
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        # Use relative improvement
        relative_improvement = (self.best_loss - val_loss) / self.best_loss
        if val_loss < self.best_loss and relative_improvement > self.min_delta_pct / 100:
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

def preprocess_epoch_sdfs_cached(unique_scenes, dataset, device):
    """
    Pre-LOAD (not encode) unique SDFs for the epoch to avoid I/O bottleneck.
    """
    print(f"Pre-loading {len(unique_scenes)} unique SDFs...")
    sdf_cache = {}
    
    for scene_idx in tqdm(unique_scenes, desc="Loading SDFs"):
        sdf = dataset.get_sdf(scene_idx).to(device)  # Load to GPU
        sdf_cache[scene_idx] = sdf
    
    return sdf_cache

def create_fast_sdf_lookup_tensor(sdf_cache):
    """
    Convert SDF cache to tensor for ultra-fast lookup.
    """
    max_scene_idx = max(sdf_cache.keys())
    sdf_shape = next(iter(sdf_cache.values())).shape
    device = next(iter(sdf_cache.values())).device
    
    # Create lookup tensor: [max_scene_idx+1, 48, 48, 48]
    lookup_tensor = torch.zeros(max_scene_idx + 1, *sdf_shape, device=device)
    
    for scene_idx, sdf in sdf_cache.items():
        lookup_tensor[scene_idx] = sdf
    
    return lookup_tensor

def process_batch_fast_with_encoding(batch, sdf_lookup_tensor, model, device):
    """
    Ultra-fast batch processing with SDF encoding in forward pass.
    """
    scene_indices = batch['scene_idx'].to(device)
    grasp_batch = batch['grasp'].to(device)
    score_batch = batch['score'].to(device)
    
    # Ultra-fast SDF lookup
    sdf_batch = sdf_lookup_tensor[scene_indices]  # [batch_size, 48, 48, 48]
    
    # Encode SDFs (gradients flow!)
    sdf_features_batch = model.encode_sdf(sdf_batch)  # [batch_size, feature_dim]
    
    return sdf_features_batch, grasp_batch, score_batch

def create_feature_lookup_tensor(scene_to_features, max_scene_idx):
    """
    Convert feature dict to tensor for ultra-fast vectorized lookup.
    """
    feature_dim = next(iter(scene_to_features.values())).shape[0]
    device = next(iter(scene_to_features.values())).device
    
    # Create lookup tensor
    lookup_tensor = torch.zeros(max_scene_idx + 1, feature_dim, device=device)
    
    for scene_idx, features in scene_to_features.items():
        lookup_tensor[scene_idx] = features
    
    return lookup_tensor

def process_batch_vectorized(batch, feature_lookup_tensor, device):
    """
    Ultra-fast vectorized batch processing.
    """
    scene_indices = batch['scene_idx'].to(device)
    grasp_batch = batch['grasp'].to(device, non_blocking=True)
    score_batch = batch['score'].to(device, non_blocking=True)
    
    # Vectorized lookup - no Python loops!
    sdf_features_batch = feature_lookup_tensor[scene_indices]
    
    return sdf_features_batch, grasp_batch, score_batch

def encode_unique_sdfs_for_epoch_efficient(unique_scenes, dataset, model, device, batch_size=64):
    """
    Memory-efficient SDF encoding with tensor conversion.
    """
    unique_scenes_list = list(unique_scenes)
    scene_to_features = {}
    
    print(f"Encoding {len(unique_scenes_list)} unique SDFs in batches of {batch_size}...")
    
    with torch.no_grad() if not model.training else torch.enable_grad():
        for i in tqdm(range(0, len(unique_scenes_list), batch_size), desc="Encoding unique SDFs"):
            batch_scenes = unique_scenes_list[i:i + batch_size]
            
            # Load batch of SDFs
            sdf_batch = torch.stack([dataset.get_sdf(scene_idx) for scene_idx in batch_scenes])
            sdf_batch = sdf_batch.to(device, non_blocking=True)
            
            # Encode batch
            if model.training:
                sdf_features_batch = model.encode_sdf(sdf_batch)
            else:
                with torch.no_grad():
                    sdf_features_batch = model.encode_sdf(sdf_batch)
            
            # Store features
            for j, scene_idx in enumerate(batch_scenes):
                scene_to_features[scene_idx] = sdf_features_batch[j] if model.training else sdf_features_batch[j].detach()
            
            del sdf_batch, sdf_features_batch
    
    # Convert to fast lookup tensor
    max_scene_idx = max(unique_scenes_list)
    feature_lookup_tensor = create_feature_lookup_tensor(scene_to_features, max_scene_idx)
    del scene_to_features  # Free memory
    
    return feature_lookup_tensor

def encode_sdfs_batch_wise(scene_indices, dataset, model, device):
    """Encode SDFs for a batch of scenes with gradient flow."""
    unique_scenes = torch.unique(scene_indices)
    
    # Load unique SDFs
    unique_sdfs = torch.stack([dataset.get_sdf(idx.item()) for idx in unique_scenes]).to(device)
    
    # Encode unique SDFs
    unique_features = model.encode_sdf(unique_sdfs)
    
    # Create mapping from scene_idx to feature_idx
    scene_to_feature_idx = {scene.item(): i for i, scene in enumerate(unique_scenes)}
    
    # Map back to original batch order
    feature_indices = [scene_to_feature_idx[idx.item()] for idx in scene_indices]
    batch_features = unique_features[feature_indices]
    
    return batch_features

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

    # Create DataLoaders with MORE WORKERS for parallel loading
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,  # Use configurable workers
        pin_memory=pin_memory, 
        persistent_workers=True,
        shuffle=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,  # Use configurable workers
        pin_memory=pin_memory, 
        persistent_workers=True,
        shuffle=False,
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

    # Remove the memory-intensive one-time preprocessing!
    # Go directly to training loop

    # --- Training Loop ---
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # === TRAINING (Encode SDFs fresh for each batch) ===
        model.train()
        total_train_loss = 0
        num_train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for batch in pbar:
            optimizer.zero_grad()

            # Get scene indices and data
            scene_indices = batch['scene_idx'].to(device)
            grasp_batch = batch['grasp'].to(device, non_blocking=True)
            score_batch = batch['score'].to(device, non_blocking=True)
            
            # Load and encode SDFs fresh for each batch (gradients flow!)
            sdf_batch = torch.stack([dataset.get_sdf(idx.item()) for idx in scene_indices]).to(device)
            sdf_features_batch = model.encode_sdf(sdf_batch)

            # Forward pass
            flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
            pred_quality = model(flattened_features)
            loss = criterion(pred_quality, score_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * sdf_features_batch.size(0)
            num_train_steps += sdf_features_batch.size(0)
            
            running_loss = total_train_loss / num_train_steps
            pbar.set_postfix(loss=f'{running_loss:.4f}')

        avg_train_loss = total_train_loss / num_train_steps

        # === VALIDATION (Use pre-computed features for efficiency) ===
        model.eval()
        start_time = time.time()
        val_feature_lookup = encode_unique_sdfs_for_epoch_efficient(
            scene_mappings['val_scenes'], dataset, model, device, batch_size=args.sdf_batch_size
        )
        encoding_time = time.time() - start_time
        
        total_val_loss = 0
        num_val_steps = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for batch in pbar_val:
                sdf_features_batch, grasp_batch, score_batch = process_batch_vectorized(
                    batch, val_feature_lookup, device
                )

                flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
                pred_quality = model(flattened_features)
                loss = criterion(pred_quality, score_batch)

                total_val_loss += loss.item() * sdf_features_batch.size(0)
                num_val_steps += sdf_features_batch.size(0)
                
                running_val_loss = total_val_loss / num_val_steps
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_val_steps
        
        # Clear validation tensors
        del val_feature_lookup
        
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
            "preprocessing_time": encoding_time # Changed from preprocessing_time to encoding_time
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