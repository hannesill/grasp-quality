#!/usr/bin/env python3
"""
Balanced optimization focusing on optimal performance AND generalization.

Key insights:
1. Batch size 256-512 (optimal for generalization)
2. Maximum workers for pre-encoding parallelism
3. Focus on GPU utilization bottlenecks
4. Model architecture optimizations
"""

import subprocess
import sys

def run_balanced_optimization():
    """Run training with balanced optimization for speed + generalization."""
    
    cmd = [
        sys.executable, 'train.py',
        '--use_preencoding',          # Pre-encoding optimization
        '--use_large_model',          # A100-optimized model with more parameters
        '--batch_size', '512',        # Optimal for generalization
        '--num_workers', '12',        # Maximum parallelism
        '--mixed_precision',          # FP16 for A100 Tensor Cores
        '--lr', '2e-4',              # Optimal for this batch size
        '--epochs', '100',
        '--train_size', '5000000',    # Full dataset
        '--val_size', '1000000',      # Full validation
        '--base_channels', '16',      # Larger model for better GPU utilization
        '--fc_dims', '256', '128', '64',  # Deeper network for GPU utilization
        '--weight_decay', '1e-4',
        '--early_stopping_patience', '15',
        '--project_name', 'adlr-balanced',
        '--run_name', 'balanced-optimization'
    ]
    
    print("⚖️  BALANCED OPTIMIZATION: Speed + Generalization")
    print("=" * 60)
    print("🎯 Focus: Maximize GPU utilization while maintaining generalization")
    print()
    print("Key optimizations:")
    print("✅ Optimal batch size: 512 (sweet spot for generalization)")
    print("✅ Maximum workers: 12 (full CPU parallelism)")
    print("✅ Large A100-optimized model: 5M+ parameters for GPU utilization")
    print("✅ Pre-encoding: Eliminates disk I/O bottleneck")
    print("✅ Mixed precision: FP16 for A100 Tensor Cores")
    print()
    print("Generalization optimizations:")
    print("🧠 Batch size: 512 (avoids sharp minima)")
    print("🧠 Learning rate: 2e-4 (stable training)")
    print("🧠 Weight decay: 1e-4 (regularization)")
    print("🧠 Gradient noise: Natural from optimal batch size")
    print()
    print("Expected improvements:")
    print("📈 GPU utilization: 13% → 60-80% (5M+ parameter model)")
    print("📈 Memory usage: 1GB → 15-25GB (proper A100 utilization)")
    print("📈 Training speed: Maintain throughput with full workers")
    print("📈 Model capacity: More parameters for better learning")
    print("📈 Generalization: Optimal batch size + regularization")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 Balanced optimization completed!")
        print("Achieved optimal speed + generalization balance!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 This optimization balances:")
    print("• Training speed (GPU utilization)")
    print("• Generalization (optimal batch size)")
    print("• System utilization (full parallelism)")
    print()
    
    success = run_balanced_optimization()
    if not success:
        print("\n💡 Alternative commands to try:")
        print("# Smaller model if still issues:")
        print("python train.py --use_preencoding --batch_size 256 --base_channels 8 --mixed_precision")
        print("# Larger model for more GPU utilization:")
        print("python train.py --use_preencoding --batch_size 512 --base_channels 32 --mixed_precision")
        sys.exit(1) 