import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from pathlib import Path
import random
import math

class GraspDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        scene_file = self.data_files[idx]
        with np.load(scene_file) as scene_data:
            # Only use the last 7 entries in each grasp, as they are the values of the hand pose
            grasps = scene_data["grasps"][:, -7:]    # shape: (N=480, G_dim=7)
            sdf = scene_data["sdf"]
            scores = scene_data["scores"]    # shape: (N=480,)

        # Pad to exactly 480 grasp-score pairs (some scenes have 476-479 grasps)
        num_grasps = grasps.shape[0]
        if num_grasps < 480:
            pad_size = 480 - num_grasps
            indices_to_duplicate = np.random.choice(num_grasps, pad_size, replace=True)
            grasps_pad = grasps[indices_to_duplicate]
            scores_pad = scores[indices_to_duplicate]
            grasps = np.concatenate([grasps, grasps_pad], axis=0)
            scores = np.concatenate([scores, scores_pad], axis=0)
        elif num_grasps > 480:
            raise ValueError(f"Scene {scene_file} has more than 480 grasps ({num_grasps}) — this is unexpected.")

        # Final assert to ensure safety
        assert grasps.shape[0] == 480
        assert scores.shape[0] == 480

        # Convert to float tensors
        scene_tensors = {
            "sdf": torch.from_numpy(sdf).float(),
            "grasps": torch.from_numpy(grasps).float(),
            "scores": torch.from_numpy(scores).float()
        }
        
        return scene_tensors


class GraspBatchIterableDataset(IterableDataset):
    def __init__(self, scene_dataset, grasp_batch_size=32, shuffle_scenes=True):
        super().__init__()
        self.scene_dataset = scene_dataset
        self.grasp_batch_size = grasp_batch_size
        self.shuffle_scenes = shuffle_scenes

    def _iter_scenes(self):
        worker_info = torch.utils.data.get_worker_info()
        
        scene_indices = list(range(len(self.scene_dataset)))
        if self.shuffle_scenes:
            random.shuffle(scene_indices)

        if worker_info is None:
            # single-process data loading, iterate over all scenes
            indices_to_process = scene_indices
        else:
            # multi-process data loading, split the work
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(scene_indices) / float(num_workers)))
            start = worker_id * per_worker
            end = min(start + per_worker, len(scene_indices))
            indices_to_process = scene_indices[start:end]

        for scene_idx in indices_to_process:
             yield self.scene_dataset[scene_idx]

    def __iter__(self):
        for scene in self._iter_scenes():
            sdf = scene['sdf']
            grasps = scene['grasps']
            scores = scene['scores']
            
            num_grasps = grasps.shape[0]
            if num_grasps == 0:
                continue

            perm = torch.randperm(num_grasps)
            shuffled_grasps = grasps[perm]
            shuffled_scores = scores[perm]

            for i in range(0, num_grasps, self.grasp_batch_size):
                grasp_batch = shuffled_grasps[i : i + self.grasp_batch_size]
                score_batch = shuffled_scores[i : i + self.grasp_batch_size]
                
                yield sdf, grasp_batch, score_batch