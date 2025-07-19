import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm


class GraspDataset(Dataset):
    def __init__(self, data_path, preload=True):
        super().__init__()
        self.preload = preload
        self.data_path = data_path

        print("Collecting data paths...")
        self.data_files = sorted([dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()])
        print(f"Found {len(self.data_files)} scenes")

        if self.preload:
            self.sdfs = []
            self.translations = []
            self.scales = []
            self.grasps = []
            self.scores = []
            for file in tqdm(self.data_files, desc="Loading scenes"):
                with np.load(file) as scene_data:
                    self.sdfs.append(scene_data["sdf"])
                    self.translations.append(scene_data["translation"])
                    self.scales.append(scene_data["global_scale"])
                    self.grasps.append(scene_data["grasps"])
                    self.scores.append(scene_data["scores"])

        # each scene has 480 grasps
        self.grasp_locations = [(scene_idx, grasp_idx) for scene_idx in range(len(self.data_files)) for grasp_idx in range(480)]

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

        # Handle IndexError by selecting a random valid grasp
        num_grasps = grasps.shape[0]
        if grasp_idx >= num_grasps:
            grasp_idx = np.random.randint(num_grasps)

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
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                