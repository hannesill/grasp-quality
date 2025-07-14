# Result Analysis

## Training optimizations

14.07.

python train.py --num_workers 12 --train_size 5000 --val_size 1000 --lr 1e-3 --batch_size 256 --mixed_precision

=== EPOCH 1 TIMING BREAKDOWN ===
Total epoch time: 28.84s
Training time: 22.94s (79.5%)
Validation time: 5.90s (20.5%)

Training timing breakdown:
  Data loading: 0.00s (0.0%)
  SDF loading: 20.12s (87.7%)
  Forward pass (inc. SDF encoding): 0.42s (1.9%)
  Backward pass: 0.58s (2.5%)