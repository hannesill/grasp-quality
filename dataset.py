import torch
from torch.utils.data import Dataset
import numpy as np
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

        # Only use the last 7 entries in each grasp, as they are the values of the hand pose
        grasps = scene_data["grasps"][:, -7:]

        # Convert to tensors
        scene_tensors = {
            "sdf": torch.from_numpy(scene_data["sdf"]),
            "grasps": torch.from_numpy(grasps),
            "scores": torch.from_numpy(scene_data["scores"])
        }
        
        return scene_tensors
