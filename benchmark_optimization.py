import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from dataset import GraspDataset, OptimizedGraspDataset
from model import GQEstimator

def benchmark_training_approaches():
    """
    Benchmark the old vs new training approach to measure speedup.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = Path('data/processed')
    
    # Test parameters
    batch_size = 32
    num_batches = 10
    
    print(f"Benchmarking training approaches on {device}")
    print(f"Batch size: {batch_size}, Number of batches: {num_batches}")
    
    # Initialize model
    model = GQEstimator(input_size=48, base_channels=4, fc_dims=[32, 16]).to(device)
    model.eval()  # Use eval mode for consistent timing
    
    # --- Test Original Approach ---
    print("\n=== Original Approach ===")
    original_dataset = GraspDataset(data_path)
    original_subset = Subset(original_dataset, list(range(min(1000, len(original_dataset)))))
    original_loader = DataLoader(original_subset, batch_size=batch_size, shuffle=False)
    
    start_time = time.time()
    total_sdf_encodings = 0
    
    with torch.no_grad():
        for i, batch in enumerate(original_loader):
            if i >= num_batches:
                break
                
            sdf_batch = batch['sdf'].to(device)
            grasp_batch = batch['grasp'].to(device)
            
            # Count SDF encodings
            total_sdf_encodings += sdf_batch.size(0)
            
            # Original approach: encode all SDFs
            sdf_features = model.encode_sdf(sdf_batch)
            flattened_features = torch.cat([sdf_features, grasp_batch], dim=1)
            pred_quality = model(flattened_features)
    
    original_time = time.time() - start_time
    print(f"Time: {original_time:.3f}s")
    print(f"Total SDF encodings: {total_sdf_encodings}")
    
    # --- Test Optimized Approach ---
    print("\n=== Optimized Approach ===")
    optimized_dataset = OptimizedGraspDataset(data_path)
    optimized_subset = Subset(optimized_dataset, list(range(min(1000, len(optimized_dataset)))))
    optimized_loader = DataLoader(optimized_subset, batch_size=batch_size, shuffle=False)
    
    def process_batch_optimized(batch, dataset, model, device):
        scene_indices = batch['scene_idx']
        grasp_batch = batch['grasp'].to(device)
        
        unique_scene_indices = torch.unique(scene_indices)
        
        sdf_features_map = {}
        for scene_idx in unique_scene_indices:
            sdf = dataset.dataset.get_sdf(scene_idx.item()).to(device)  # dataset.dataset because of Subset wrapper
            sdf_features = model.encode_sdf(sdf)
            sdf_features_map[scene_idx.item()] = sdf_features
        
        batch_size = len(scene_indices)
        feature_dim = next(iter(sdf_features_map.values())).shape[0]
        sdf_features_batch = torch.zeros(batch_size, feature_dim, device=device)
        
        for i, scene_idx in enumerate(scene_indices):
            sdf_features_batch[i] = sdf_features_map[scene_idx.item()]
        
        return sdf_features_batch, grasp_batch, len(unique_scene_indices)
    
    start_time = time.time()
    total_unique_sdf_encodings = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(optimized_loader):
            if i >= num_batches:
                break
            
            sdf_features_batch, grasp_batch, unique_count = process_batch_optimized(
                batch, optimized_subset, model, device
            )
            
            total_unique_sdf_encodings += unique_count
            total_samples += len(batch['scene_idx'])
            
            flattened_features = torch.cat([sdf_features_batch, grasp_batch], dim=1)
            pred_quality = model(flattened_features)
    
    optimized_time = time.time() - start_time
    print(f"Time: {optimized_time:.3f}s")
    print(f"Total unique SDF encodings: {total_unique_sdf_encodings}")
    print(f"Total samples: {total_samples}")
    print(f"Duplicate encodings avoided: {total_samples - total_unique_sdf_encodings}")
    
    # --- Results ---
    print("\n=== Results ===")
    speedup = original_time / optimized_time
    efficiency = total_unique_sdf_encodings / total_sdf_encodings
    print(f"Speedup: {speedup:.2f}x")
    print(f"SDF encoding efficiency: {efficiency:.2f} (lower is better)")
    print(f"Redundant encodings eliminated: {(1 - efficiency) * 100:.1f}%")

if __name__ == "__main__":
    benchmark_training_approaches() 