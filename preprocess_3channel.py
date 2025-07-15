import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import argparse

from hand_kinematics import HandKinematics

def process_single_scene(args):
    """Process a single scene file to generate 3-channel inputs."""
    scene_file, output_dir, hand_kinematics_urdf = args
    
    try:
        # Create hand kinematics for this process
        hand = HandKinematics(hand_kinematics_urdf)
        
        # Load scene data
        with np.load(scene_file) as scene_data:
            sdf = scene_data["sdf"]  # (48, 48, 48)
            grasps = scene_data["grasps"]  # (N, 19)
            scores = scene_data["scores"]  # (N,)
        
        num_grasps = grasps.shape[0]
        
        # Pre-allocate output arrays
        sdf_channels = np.zeros((num_grasps, 48, 48, 48))
        palm_distance_channels = np.zeros((num_grasps, 48, 48, 48))
        fingertip_distance_channels = np.zeros((num_grasps, 48, 48, 48))
        
        # Process each grasp
        for i in range(num_grasps):
            grasp = grasps[i]
            
            # Set SDF channel (same for all grasps in this scene)
            sdf_channels[i] = sdf
            
            # Compute distance fields for this grasp
            palm_dist, fingertip_dist = hand.compute_distance_fields(grasp)
            palm_distance_channels[i] = palm_dist
            fingertip_distance_channels[i] = fingertip_dist
        
        # Stack channels: (N, 3, 48, 48, 48)
        three_channel_inputs = np.stack([
            sdf_channels,
            palm_distance_channels,
            fingertip_distance_channels
        ], axis=1)
        
        # Save preprocessed data
        scene_name = scene_file.stem
        output_file = output_dir / f"{scene_name}_3channel.npz"
        
        np.savez_compressed(
            output_file,
            inputs=three_channel_inputs,  # (N, 3, 48, 48, 48)
            grasps=grasps,               # (N, 19)
            scores=scores                # (N,)
        )
        
        return f"✅ Processed {scene_name}: {num_grasps} grasps"
        
    except Exception as e:
        return f"❌ Error processing {scene_file}: {e}"

def preprocess_dataset(data_path, output_path, num_workers=8):
    """
    Preprocess the entire dataset to generate 3-channel inputs.
    
    Args:
        data_path: Path to the original dataset
        output_path: Path to save preprocessed data
        num_workers: Number of parallel workers
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    # Find all scene files
    scene_files = [dir / 'scene.npz' for dir in data_path.iterdir() 
                   if dir.is_dir() and (dir / 'scene.npz').exists()]
    
    print(f"📁 Found {len(scene_files)} scenes to process")
    print(f"🔧 Using {num_workers} parallel workers")
    print(f"💾 Output directory: {output_path}")
    
    # Prepare arguments for parallel processing
    process_args = [(scene_file, output_path, "urdfs/dlr2.urdf") 
                    for scene_file in scene_files]
    
    # Process scenes in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_scene = {executor.submit(process_single_scene, arg): arg[0] 
                          for arg in process_args}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_scene), total=len(scene_files), 
                          desc="Processing scenes"):
            result = future.result()
            if result.startswith("✅"):
                successful += 1
            else:
                failed += 1
                print(result)
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 PREPROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {successful} scenes")
    print(f"❌ Failed: {failed} scenes")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🚀 Average time per scene: {total_time/len(scene_files):.2f}s")

def create_gpu_cached_dataset(preprocessed_path, device='cuda'):
    """
    Create a GPU-cached version of the preprocessed dataset.
    
    Args:
        preprocessed_path: Path to preprocessed data
        device: Device to load data to
    """
    preprocessed_path = Path(preprocessed_path)
    
    # Find all preprocessed files
    preprocessed_files = list(preprocessed_path.glob("*_3channel.npz"))
    
    print(f"📁 Found {len(preprocessed_files)} preprocessed files")
    print(f"🔥 Loading all data to {device}...")
    
    # Load all data
    all_inputs = []
    all_grasps = []
    all_scores = []
    
    for file in tqdm(preprocessed_files, desc="Loading preprocessed data"):
        with np.load(file) as data:
            inputs = data['inputs']  # (N, 3, 48, 48, 48)
            grasps = data['grasps']  # (N, 19)
            scores = data['scores']  # (N,)
            
            all_inputs.append(inputs)
            all_grasps.append(grasps)
            all_scores.append(scores)
    
    # Stack all data
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_grasps = np.concatenate(all_grasps, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    print(f"📊 Total dataset size: {len(all_inputs)} samples")
    print(f"📐 Input shape: {all_inputs.shape}")
    print(f"💾 Memory usage: {all_inputs.nbytes / (1024**3):.2f} GB")
    
    # Convert to tensors and move to device
    if device == 'cuda' and torch.cuda.is_available():
        inputs_tensor = torch.from_numpy(all_inputs).float().to(device)
        grasps_tensor = torch.from_numpy(all_grasps).float().to(device)
        scores_tensor = torch.from_numpy(all_scores).float().to(device)
        
        print(f"🚀 Data loaded to GPU successfully!")
        print(f"📊 GPU memory usage: {inputs_tensor.numel() * 4 / (1024**3):.2f} GB")
        
        return inputs_tensor, grasps_tensor, scores_tensor
    else:
        print("⚠️  CUDA not available, keeping data on CPU")
        return all_inputs, all_grasps, all_scores

def main():
    parser = argparse.ArgumentParser(description="Preprocess grasp dataset to 3-channel inputs")
    parser.add_argument('--data_path', type=str, default='data/processed', 
                       help='Path to original dataset')
    parser.add_argument('--output_path', type=str, default='data/preprocessed_3channel',
                       help='Path to save preprocessed data')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--create_gpu_cache', action='store_true',
                       help='Create GPU-cached version after preprocessing')
    
    args = parser.parse_args()
    
    # Step 1: Preprocess dataset
    print("🔄 Starting dataset preprocessing...")
    preprocess_dataset(args.data_path, args.output_path, args.num_workers)
    
    # Step 2: Create GPU cache if requested
    if args.create_gpu_cache:
        print("\n🔄 Creating GPU-cached dataset...")
        inputs, grasps, scores = create_gpu_cached_dataset(args.output_path)
        
        # Save GPU cache
        cache_file = Path(args.output_path) / "gpu_cache.pth"
        torch.save({
            'inputs': inputs,
            'grasps': grasps,
            'scores': scores
        }, cache_file)
        print(f"💾 GPU cache saved to {cache_file}")

if __name__ == "__main__":
    main() 