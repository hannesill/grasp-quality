import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt


# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import GraspDataset
from src.model import GQEstimator


class GraspOnlyDataset(Dataset):
    """Wraps GraspDataset to return only lightweight fields (no SDF tensors)."""

    def __init__(self, base_dataset: GraspDataset):
        super().__init__()
        if not getattr(base_dataset, 'preload', False):
            raise ValueError("GraspOnlyDataset requires base_dataset.preload=True for fast access.")
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base.grasp_locations)

    def __getitem__(self, idx: int):
        scene_idx, grasp_idx = self.base.grasp_locations[idx]
        grasp = self.base.grasps[scene_idx][grasp_idx]
        score = self.base.scores[scene_idx][grasp_idx]
        return {
            'grasp': torch.from_numpy(grasp).float(),
            'score': torch.tensor(score, dtype=torch.float32),
            'scene_idx': scene_idx,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GQ model: sample 100k pairs, predict, and plot targets vs predictions sorted by target.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data root')
    parser.add_argument('--splits_file', type=str, default='data/splits.json', help='Path to splits JSON file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate (default: test)')
    parser.add_argument('--num_samples', type=int, default=100_000, help='Number of random (grasp, score) pairs to evaluate (default: 100000)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for prediction (default: 512)')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels used in the model (must match training)')
    parser.add_argument('--fc_dims', nargs='+', type=int, default=[256, 128, 64], help='FC dims used in the model head (must match training)')
    parser.add_argument('--output', type=str, default='plots/gq_ordering_final.png', help='Output path for the plot PNG')
    return parser.parse_args()


@torch.no_grad()
def precompute_sdf_features(model: GQEstimator, dataset: GraspDataset, device: torch.device) -> torch.Tensor:
    """Encode each scene's SDF once using the model's object_encoder.

    Returns tensor of shape (num_scenes, flattened_dim) on device.
    """
    model.eval()
    object_encoder = model.object_encoder
    object_encoder.eval()

    num_scenes = len(dataset.sdfs)

    # Determine flattened feature dim by a dry run on one SDF
    sample_sdf = torch.from_numpy(dataset.sdfs[0]).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,D,D)
    sample_feat = object_encoder(sample_sdf).view(1, -1)
    feat_dim = sample_feat.shape[1]

    features = torch.empty((num_scenes, feat_dim), dtype=sample_feat.dtype, device=device)

    for i in range(num_scenes):
        sdf = torch.from_numpy(dataset.sdfs[i]).float().unsqueeze(0).unsqueeze(0).to(device)
        feat = object_encoder(sdf).view(1, -1)
        features[i] = feat[0]

    return features


def sample_indices(total: int, n: int) -> List[int]:
    if total >= n:
        return random.sample(range(total), n)
    # sample with replacement to reach n
    return [random.randrange(total) for _ in range(n)]


def evaluate_and_plot(args):
    device = torch.device(args.device)

    # Dataset
    data_path = Path(args.data_path)
    base_dataset = GraspDataset(data_path, split=args.split, splits_file=args.splits_file, preload=True)
    eval_dataset = GraspOnlyDataset(base_dataset)

    # Model
    model = GQEstimator(input_size=48, base_channels=args.base_channels, fc_dims=args.fc_dims).to(device)
    raw_state = torch.load(args.model_path, map_location=device)

    # Extract state_dict if wrapped
    state_dict = raw_state.get('state_dict', raw_state) if isinstance(raw_state, dict) else raw_state

    # Remap keys to match current model naming
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[len('module.') :]
        if new_k.startswith('_orig_mod.'):
            new_k = new_k[len('_orig_mod.') :]
        # Older checkpoints used 'conv_block' for the encoder sequential
        if new_k.startswith('conv_block.'):
            new_k = new_k.replace('conv_block.', 'object_encoder.layers.', 1)
        # Some checkpoints may already have 'object_encoder.layers.' or 'gq_head.' which is fine
        remapped[new_k] = v

    model_sd = model.state_dict()
    loadable = {k: v for k, v in remapped.items() if k in model_sd and model_sd[k].shape == v.shape}
    missing = [k for k in model_sd.keys() if k not in loadable]
    skipped = [k for k in remapped.keys() if k not in loadable]

    print(f"Loading {len(loadable)}/{len(model_sd)} parameters from checkpoint; skipping {len(skipped)}")
    model.load_state_dict(loadable, strict=False)
    model.eval()

    # Precompute SDF features once per scene
    sdf_features_cache = precompute_sdf_features(model, base_dataset, device=device)

    # Randomly sample indices
    total = len(eval_dataset)
    indices = sample_indices(total, args.num_samples)
    subset = Subset(eval_dataset, indices)

    pin_memory = device.type == 'cuda'
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    all_targets: List[float] = []
    all_preds: List[float] = []

    with torch.no_grad():
        for batch in loader:
            grasp = batch['grasp'].to(device)  # (B, 19)
            score = batch['score'].to(device)  # (B,)
            scene_idx_tensor = torch.as_tensor(batch['scene_idx'], device=device, dtype=torch.long)

            sdf_feat = sdf_features_cache[scene_idx_tensor]  # (B, F)
            combined = torch.cat([sdf_feat, grasp], dim=1)
            pred = model(combined)  # (B,)

            all_targets.append(score.detach().cpu())
            all_preds.append(pred.detach().cpu())

    targets = torch.cat(all_targets).numpy()
    preds = torch.cat(all_preds).numpy()

    # Sort by target
    order = np.argsort(targets)
    targets_sorted = targets[order]
    preds_sorted = preds[order]

    # Plot
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.figure(figsize=(12, 5), dpi=200)
    x = np.arange(len(targets_sorted))
    plt.plot(x, targets_sorted, label='Target', linewidth=1.0)
    plt.plot(x, preds_sorted, label='Predicted', linewidth=1.0, alpha=0.85)
    plt.xlabel('Samples (sorted by target)')
    plt.ylabel('Grasp score')
    plt.title(f'Predicted vs Target Scores (N={len(targets_sorted):,}, split={args.split})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()

    # Also print simple stats
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    corr = float(np.corrcoef(targets, preds)[0, 1]) if len(targets) > 1 else float('nan')
    print(f"Saved plot to {args.output}")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, Corr: {corr:.4f}")


if __name__ == '__main__':
    args = parse_args()
    evaluate_and_plot(args)


