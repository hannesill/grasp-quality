import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

class SceneDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        scene_file = self.data_files[idx]
        with np.load(scene_file) as scene_data:
            grasps = scene_data["grasps"]    # shape: (N=480, G_dim=19)
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
    

class GraspDataset(Dataset):
    def __init__(self, data_path, shuffle_grasps=True):
        super().__init__()
        self.data_path = data_path
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()]

        self.shuffle_grasps = shuffle_grasps

        # Assume each scene has 480 grasps & create list of (scene_idx, grasp_idx) tuples
        self.grasp_locations = [(scene_idx, grasp_idx) for scene_idx in range(len(self.data_files)) for grasp_idx in range(480)]

        if self.shuffle_grasps:
            self.grasp_indices = torch.randperm(len(self.grasp_locations))
        else:
            self.grasp_indices = list(range(len(self.grasp_locations)))

    def __len__(self):
        return len(self.grasp_locations)

    def __getitem__(self, idx):
        scene_idx, grasp_idx = self.grasp_locations[idx]
        with np.load(self.data_files[scene_idx]) as scene_data:
            sdf = scene_data["sdf"]
            grasps = scene_data["grasps"][:, :]    # shape: (N=480, G_dim=19)
            scores = scene_data["scores"]    # shape: (N=480,)

        # Handle IndexError by selecting a random valid grasp
        num_grasps = grasps.shape[0]
        if grasp_idx >= num_grasps:
            grasp_idx = np.random.randint(num_grasps)

        grasp = grasps[grasp_idx]
        score = scores[grasp_idx]

        return {
            "sdf": torch.from_numpy(sdf).float(),
            "grasp": torch.from_numpy(grasp).float(),
            "score": torch.tensor(score, dtype=torch.float32),
            "scene_idx": scene_idx,
            "grasp_idx": grasp_idx
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                

class OptimizedGraspDataset(Dataset):
    """
    Optimized dataset that returns grasp-score pairs with scene indices instead of full SDFs.
    This allows preprocessing unique SDFs only once per epoch during training.
    """
    def __init__(self, data_path, shuffle_grasps=True):
        super().__init__()
        self.data_path = data_path
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()]
        self.shuffle_grasps = shuffle_grasps

        # Create list of (scene_idx, grasp_idx) tuples
        self.grasp_locations = [(scene_idx, grasp_idx) for scene_idx in range(len(self.data_files)) for grasp_idx in range(480)]

        if self.shuffle_grasps:
            self.grasp_indices = torch.randperm(len(self.grasp_locations))
        else:
            self.grasp_indices = list(range(len(self.grasp_locations)))

        # Cache for loaded scenes to avoid repeated file I/O
        self._scene_cache = {}
        self._cache_size_limit = 100  # Increase cache size since we're more efficient now

    def _load_scene(self, scene_idx):
        """Load scene data with caching."""
        if scene_idx in self._scene_cache:
            return self._scene_cache[scene_idx]
        
        # If cache is full, clear oldest entries (simple FIFO strategy)
        if len(self._scene_cache) >= self._cache_size_limit:
            # Remove oldest half of cache
            keys_to_remove = list(self._scene_cache.keys())[:len(self._scene_cache)//2]
            for key in keys_to_remove:
                del self._scene_cache[key]
        
        with np.load(self.data_files[scene_idx]) as scene_data:
            scene = {
                'sdf': torch.from_numpy(scene_data["sdf"]).float(),
                'grasps': torch.from_numpy(scene_data["grasps"]).float(),
                'scores': torch.from_numpy(scene_data["scores"]).float()
            }
        
        self._scene_cache[scene_idx] = scene
        return scene

    def get_sdf(self, scene_idx):
        """Get SDF for a specific scene index."""
        scene = self._load_scene(scene_idx)
        return scene['sdf']

    def __len__(self):
        return len(self.grasp_locations)

    def __getitem__(self, idx):
        scene_idx, grasp_idx = self.grasp_locations[idx]
        scene = self._load_scene(scene_idx)
        
        # Handle IndexError by selecting a random valid grasp
        num_grasps = scene['grasps'].shape[0]
        if grasp_idx >= num_grasps:
            grasp_idx = np.random.randint(num_grasps)

        return {
            "scene_idx": scene_idx,
            "grasp": scene['grasps'][grasp_idx],
            "score": scene['scores'][grasp_idx],
            "grasp_idx": grasp_idx
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                