# Result Analysis

## Training optimizations

14.07.

### Before Preencoding SDFs
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

### After Preencoding SDFs
Train dataset size: 5000, Validation dataset size: 1000

Pre-encoding completed in 21.55s
Epoch 1/100 Training: 100%|███████████████████████████████████████████████████████| 78/78 [00:04<00:00, 17.03it/s, batch_time=0.003s, loss=69.2318]
Epoch 1/100 Validation: 100%|████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 17.74it/s, val_loss=74.1633]

=== EPOCH 1 TIMING BREAKDOWN ===
Total epoch time: 27.03s
Training time: 4.58s (16.9%)
Validation time: 0.90s (3.3%)

Training timing breakdown:
  Data loading: 0.01s (0.2%)
  SDF loading: 0.00s (0.1%)
  Forward pass (inc. SDF encoding): 3.49s (76.1%)
  Backward pass: 0.82s (18.0%)

