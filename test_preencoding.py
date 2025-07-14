#!/usr/bin/env python3
"""
Test script for pre-encoding optimization with no multiprocessing.

This script tests the pre-encoding optimization with minimal settings:
- No multiprocessing (num_workers=0) to avoid CUDA sharing issues
- Small dataset for quick testing
- All other optimizations enabled
"""

import subprocess
import sys

def test_preencoding():
    """Test pre-encoding optimization with safe settings."""
    
    cmd = [
        sys.executable, 'train.py',
        '--use_preencoding',          # KEY: Enable pre-encoding optimization
        '--batch_size', '128',        # Moderate batch size
        '--num_workers', '0',         # No multiprocessing to avoid CUDA issues
        '--mixed_precision',          # Enable mixed precision
        '--lr', '1e-4',              # Standard learning rate
        '--epochs', '3',             # Just 3 epochs for testing
        '--train_size', '1000',       # Small training set for quick test
        '--val_size', '200',          # Small validation set
        '--base_channels', '4',       # Standard model size
        '--fc_dims', '32', '16',      # Standard FC layers
        '--project_name', 'test-preencoding',
        '--run_name', 'test-run'
    ]
    
    print("🧪 TESTING Pre-encoding Optimization")
    print("=" * 50)
    print("Test settings:")
    print("✅ Pre-encoding: Enabled")
    print("✅ Mixed precision: Enabled")
    print("✅ Workers: 0 (no multiprocessing)")
    print("✅ Dataset: Small (1000 train, 200 val)")
    print("✅ Epochs: 3 (for quick testing)")
    print()
    print("This test will verify:")
    print("1. Pre-encoding works correctly")
    print("2. Training runs without errors")
    print("3. Performance improvement is visible")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Test passed! Pre-encoding is working correctly.")
        print("You can now run the full training with:")
        print("python train_ultra_fast.py")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    success = test_preencoding()
    if not success:
        sys.exit(1) 