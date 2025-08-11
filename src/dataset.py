import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


class GraspDataset(Dataset):
    def __init__(self, data_path, split='train', splits_file='data/splits.json', preload=True):
        super().__init__()
        self.preload = preload
        self.data_path = Path(data_path)
        self.split = split
        self.splits_file = Path(splits_file)

        print(f"Loading data for '{self.split}' split from {self.splits_file}")

        if not self.splits_file.exists():
            raise FileNotFoundError(f"Splits file not found at {self.splits_file}. Please run scripts/create_splits.py first.")

        with open(self.splits_file, 'r') as f:
            splits = json.load(f)
        
        if self.split not in splits:
            raise ValueError(f"Split '{self.split}' not found in splits file. Available splits: {list(splits.keys())}")
        
        scene_ids = splits[self.split]
        
        print("Collecting data paths...")
        self.data_files = sorted([self.data_path / scene_id / 'scene.npz' for scene_id in scene_ids])
        
        # Check for missing files and filter them out
        original_len = len(self.data_files)
        self.data_files = [f for f in self.data_files if f.exists()]
        if len(self.data_files) != original_len:
            print(f"Warning: {original_len - len(self.data_files)} scene files not found and were skipped.")

        print(f"Found {len(self.data_files)} scenes for split '{self.split}'")

        if self.preload:
            self.sdfs = []
            self.translations = []
            self.scales = []
            self.grasps = []
            self.scores = []
            for file in tqdm(self.data_files, desc=f"Loading scenes for {self.split} split"):
                with np.load(file) as scene_data:
                    self.sdfs.append(scene_data["sdf"])
                    self.translations.append(scene_data["translation"])
                    self.scales.append(scene_data["global_scale"])
                    self.grasps.append(scene_data["grasps"])
                    self.scores.append(scene_data["scores"])

        # Create a list of (scene_idx, grasp_idx) tuples.
        # Note: Most scenes have 480 grasps, but some have slightly less.
        self.grasp_locations = []
        num_grasps_per_scene = []
        
        if self.preload:
            num_grasps_per_scene = [grasps.shape[0] for grasps in self.grasps]
        else:
            print("Scanning scenes to determine number of grasps...")
            for file in tqdm(self.data_files, desc="Scanning scenes"):
                with np.load(file, mmap_mode='r') as scene_data:
                    num_grasps_per_scene.append(scene_data['grasps'].shape[0])

        for scene_idx, num_grasps in enumerate(num_grasps_per_scene):
            for grasp_idx in range(num_grasps):
                self.grasp_locations.append((scene_idx, grasp_idx))

    def __len__(self):
        return len(self.grasp_locations)

    def _get_scene_data(self, scene_idx):
        if self.preload:
            return self.sdfs[scene_idx], self.translations[scene_idx], self.scales[scene_idx], self.grasps[scene_idx], self.scores[scene_idx]
        else:
            with np.load(self.data_files[scene_idx]) as scene_data:
                sdf = scene_data["sdf"]
                translation = scene_data["translation"]
                scale = scene_data["global_scale"]
                grasps = scene_data["grasps"]
                scores = scene_data["scores"]
                return sdf, translation, scale, grasps, scores

    def __getitem__(self, idx):
        scene_idx, grasp_idx = self.grasp_locations[idx]

        sdf, translation, scale, grasps, scores = self._get_scene_data(scene_idx)

        grasp = grasps[grasp_idx]
        score = scores[grasp_idx]

        return {
            "sdf": torch.from_numpy(sdf).float(),
            "translation": torch.tensor(translation, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "grasp": torch.from_numpy(grasp).float(),
            "score": torch.tensor(score, dtype=torch.float32),
            "scene_idx": scene_idx,
            "grasp_idx": grasp_idx,
        }
                

class GPUCachedGraspDataset(Dataset):
    """
    Ultimate performance dataset that loads ALL SDFs into GPU memory at startup.
    
    Benefits:
    - Zero data loading time during training (near-instant batch creation)
    - All SDFs stored as single contiguous GPU tensor
    - Simple indexing with no CPU-GPU transfers
    - Perfect for A100 with 40GB memory (only needs ~6.7GB for 15,547 scenes)
    """
    def __init__(self, data_path, device, split='train', splits_file='data/splits.json'):
        super().__init__()
        self.data_path = Path(data_path)
        self.device = device
        self.split = split
        self.splits_file = Path(splits_file)

        print(f"Loading data for '{self.split}' split from {self.splits_file}")

        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found at {self.splits_file}. Please run scripts/create_splits.py first."
            )

        with open(self.splits_file, 'r') as f:
            splits = json.load(f)

        if self.split not in splits:
            raise ValueError(
                f"Split '{self.split}' not found in splits file. Available splits: {list(splits.keys())}"
            )

        scene_ids = splits[self.split]

        # Collect scene file paths for this split
        self.data_files = sorted([self.data_path / scene_id / 'scene.npz' for scene_id in scene_ids])

        # Filter out missing files (be robust against partial datasets)
        original_len = len(self.data_files)
        self.data_files = [f for f in self.data_files if f.exists()]
        if len(self.data_files) != original_len:
            print(f"Warning: {original_len - len(self.data_files)} scene files not found and were skipped.")

        print(f"Found {len(self.data_files)} scenes for split '{self.split}'")
        
        # Create list of (scene_idx, grasp_idx) tuples
        # We pad to 480 grasps during load to ensure fixed-size tensors.
        self.grasp_locations = [
            (scene_idx, grasp_idx)
            for scene_idx in range(len(self.data_files))
            for grasp_idx in range(480)
        ]
        
        # Load all data into GPU memory at startup
        self.gpu_sdfs = None
        self.gpu_grasps = None
        self.gpu_scores = None
        self._load_all_to_gpu()
        
        print(f"✅ All data loaded to GPU! Memory usage: {self._get_gpu_memory_usage():.2f} GB")

    def _get_gpu_memory_usage(self):
        """Calculate GPU memory usage in GB."""
        total_bytes = 0
        if self.gpu_sdfs is not None:
            total_bytes += self.gpu_sdfs.numel() * self.gpu_sdfs.element_size()
        if self.gpu_grasps is not None:
            total_bytes += self.gpu_grasps.numel() * self.gpu_grasps.element_size()
        if self.gpu_scores is not None:
            total_bytes += self.gpu_scores.numel() * self.gpu_scores.element_size()
        return total_bytes / (1024 ** 3)

    def _load_all_to_gpu(self):
        """Load all SDFs, grasps, and scores to GPU memory."""
        print("Loading all scene data to GPU memory...")
        
        # Pre-allocate GPU tensors
        num_scenes = len(self.data_files)
        self.gpu_sdfs = torch.zeros((num_scenes, 48, 48, 48), dtype=torch.float32, device=self.device)
        self.gpu_grasps = torch.zeros((num_scenes, 480, 19), dtype=torch.float32, device=self.device)
        self.gpu_scores = torch.zeros((num_scenes, 480), dtype=torch.float32, device=self.device)
        
        # Load data with progress bar
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def load_single_scene(scene_idx):
            """Load a single scene and return data."""
            try:
                with np.load(self.data_files[scene_idx]) as scene_data:
                    sdf = torch.from_numpy(scene_data["sdf"]).float()
                    grasps = torch.from_numpy(scene_data["grasps"]).float()
                    scores = torch.from_numpy(scene_data["scores"]).float()
                    
                    # Pad to exactly 480 grasps if needed
                    num_grasps = grasps.shape[0]
                    if num_grasps < 480:
                        pad_size = 480 - num_grasps
                        indices_to_duplicate = torch.randint(0, num_grasps, (pad_size,))
                        grasps = torch.cat([grasps, grasps[indices_to_duplicate]], dim=0)
                        scores = torch.cat([scores, scores[indices_to_duplicate]], dim=0)
                    
                    return scene_idx, sdf, grasps, scores
            except Exception as e:
                print(f"Error loading scene {scene_idx}: {e}")
                return scene_idx, None, None, None
        
        # Load all scenes in parallel
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(
                executor.map(load_single_scene, range(num_scenes)),
                total=num_scenes,
                desc="Loading scenes to GPU"
            ))
        
        # Copy data to GPU tensors
        corrupted_scenes = 0
        for scene_idx, sdf, grasps, scores in results:
            if sdf is not None:
                self.gpu_sdfs[scene_idx] = sdf
                self.gpu_grasps[scene_idx] = grasps
                self.gpu_scores[scene_idx] = scores
            else:
                corrupted_scenes += 1
        
        if corrupted_scenes > 0:
            print(f"⚠️  Skipped {corrupted_scenes} corrupted scenes")
        
        print(f"✅ Loaded {num_scenes - corrupted_scenes} scenes to GPU")

    def __len__(self):
        return len(self.grasp_locations)

    def __getitem__(self, idx):
        scene_idx, grasp_idx = self.grasp_locations[idx]
        
        # Zero-copy GPU slicing - instant!
        return {
            "sdf": self.gpu_sdfs[scene_idx],  # Shape: (48, 48, 48)
            "grasp": self.gpu_grasps[scene_idx, grasp_idx],  # Shape: (19,)
            "score": self.gpu_scores[scene_idx, grasp_idx],  # Shape: ()
            "scene_idx": scene_idx,
            "grasp_idx": grasp_idx
        }
    
    def get_batch_sdfs(self, scene_indices):
        """
        Ultra-fast batch SDF retrieval with zero-copy GPU slicing.
        
        Args:
            scene_indices: Tensor of shape (batch_size,) with scene indices
        
        Returns:
            Tensor of shape (batch_size, 48, 48, 48) - direct GPU slice
        """
        return self.gpu_sdfs[scene_indices]
    
    def get_batch_data(self, scene_indices, grasp_indices):
        """
        Ultra-fast batch data retrieval.
        
        Args:
            scene_indices: Tensor of shape (batch_size,) with scene indices
            grasp_indices: Tensor of shape (batch_size,) with grasp indices
        
        Returns:
            Tuple of (sdfs, grasps, scores) - all direct GPU slices
        """
        batch_size = len(scene_indices)
        
        # Advanced indexing for grasps and scores
        grasps = self.gpu_grasps[scene_indices, grasp_indices]
        scores = self.gpu_scores[scene_indices, grasp_indices]
        sdfs = self.gpu_sdfs[scene_indices]
        
        return sdfs, grasps, scores

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                