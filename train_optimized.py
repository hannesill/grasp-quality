#!/usr/bin/env python3
"""
Optimized training script with best settings for A100 GPU.

This script demonstrates running the training with all optimizations enabled:
- GPU training with mixed precision
- Optimized batch size and workers
- Better caching and memory management
- PyTorch 2.0 compilation
"""

import subprocess
import sys

def run_optimized_training():
    """Run training with optimized settings for A100 GPU."""
    
    cmd = [
        sys.executable, 'train.py',
        '--batch_size', '128',        # Larger batch size for better GPU utilization
        '--num_workers', '8',         # Optimized for 12 vCPU system
        '--mixed_precision',          # Enable mixed precision for A100
        '--lr', '2e-4',              # Slightly higher learning rate for larger batch
        '--epochs', '100',
        '--train_size', '5000',
        '--val_size', '1000',
        '--base_channels', '8',       # Slightly larger model for better GPU utilization
        '--fc_dims', '64', '32',      # Larger FC layers
        '--weight_decay', '1e-4',
        '--early_stopping_patience', '15',
        '--project_name', 'adlr-optimized',
        '--run_name', 'optimized-training-a100'
    ]
    
    print("Running optimized training with the following settings:")
    print(" ".join(cmd))
    print("\nExpected speedup: 10x-100x compared to CPU training")
    print("Key optimizations enabled:")
    print("- GPU training with CUDA")
    print("- Mixed precision training (FP16)")
    print("- Batch SDF loading (no more loops)")
    print("- Optimized DataLoader settings")
    print("- Larger batch size for better GPU utilization")
    print("- PyTorch 2.0 compilation")
    print("- Improved memory management")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    success = run_optimized_training()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed or was interrupted.")
        sys.exit(1) 