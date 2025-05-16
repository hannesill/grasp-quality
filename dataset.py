import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
from pathlib import Path

class GraspDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = []
        
        for dir in data_path.iterdir():
            if dir.is_dir():
                scene_file = dir / 'scene.npz'
                if scene_file.exists():
                    self.data_files.append(scene_file)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        scene_file = self.data_files[idx]
        
        scene_data = np.load(scene_file)
        scene_tensors = {k: torch.from_numpy(v) for k, v in scene_data.items()}
        
        return scene_tensors
