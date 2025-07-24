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
                