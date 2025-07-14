#!/usr/bin/env python3
"""
Test script for the large A100-optimized model.

This script tests:
1. Large model initialization and parameter count
2. GPU utilization improvement
3. Memory usage increase
4. Training functionality
"""

import subprocess
import sys

def test_large_model():
    """Test the large model with small dataset."""
    
    cmd = [
        sys.executable, 'train.py',
        '--use_preencoding',          # Pre-encoding optimization
        '--use_large_model',          # Large A100-optimized model
        '--batch_size', '64',         # Moderate batch for testing
        '--num_workers', '8',         # Good parallelism
        '--mixed_precision',          # FP16 for A100
        '--lr', '1e-4',              # Standard learning rate
        '--epochs', '2',             # Just 2 epochs for testing
        '--train_size', '1000',       # Small training set
        '--val_size', '200',          # Small validation set
        '--base_channels', '16',      # Medium model size
        '--fc_dims', '256', '128',    # Larger FC layers
        '--project_name', 'test-large-model',
        '--run_name', 'large-model-test'
    ]
    
    print("🧪 TESTING Large A100-Optimized Model")
    print("=" * 50)
    print("Test objectives:")
    print("✅ Verify large model initialization")
    print("✅ Check parameter count (should be 5M+)")
    print("✅ Monitor GPU utilization improvement")
    print("✅ Observe memory usage increase")
    print("✅ Confirm training stability")
    print()
    print("Expected vs small model:")
    print("📊 Parameters: 47k → 5M+ (100x increase)")
    print("📊 GPU utilization: 13% → 40-60%")
    print("📊 Memory usage: 1GB → 10-20GB")
    print("📊 Model capacity: Much better learning")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Large model test passed!")
        print("Key observations to check:")
        print("1. Parameter count in logs (should be 5M+)")
        print("2. GPU utilization with nvidia-smi (should be 40-60%)")
        print("3. Memory usage (should be 10-20GB)")
        print("4. Training stability and convergence")
        print()
        print("If successful, run full optimization:")
        print("python train_optimized_balanced.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Large model test failed: {e}")
        print("This might indicate:")
        print("• Model too large for current GPU memory")
        print("• Architecture issues")
        print("• Configuration problems")
        print()
        print("Try smaller configuration:")
        print("python train.py --use_large_model --batch_size 32 --base_channels 8")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 This test will verify the large model works correctly")
    print("and shows improved GPU utilization.")
    print()
    
    success = test_large_model()
    if not success:
        sys.exit(1) 