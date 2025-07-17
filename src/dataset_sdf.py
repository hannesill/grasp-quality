import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm


class SDFDataset(Dataset):
    def __init__(self, data_path, preload=True):
        super().__init__()
        self.data_path = data_path
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()]
        self.preload = preload

        if self.preload:
            self.sdfs = []
            for file in tqdm(self.data_files, desc="Loading scenes"):
                with np.load(file) as scene_data:
                    self.sdfs.append(scene_data["sdf"])

        self.scene_locations = list(range(len(self.data_files)))
        np.random.shuffle(self.scene_locations)

    def __len__(self):
        return len(self.scene_locations)

    def _get_scene_data(self, scene_idx):
        if self.preload:
            return self.sdfs[scene_idx]
        else:
            with np.load(self.data_files[scene_idx]) as scene_data:
                sdf = scene_data["sdf"]
                return sdf

    def __getitem__(self, idx):
        scene_idx = self.scene_locations[idx]

        sdf = self._get_scene_data(scene_idx)

        return {
            "sdf": torch.from_numpy(sdf).float(),
            "scene_idx": scene_idx,
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                