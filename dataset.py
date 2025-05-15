import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
from pathlib import Path

class GraspDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = []
        
        # Collect all .npz and .obj files in the data_path
        for dir in data_path.iterdir():
            if dir.is_dir():
                npz_file = dir / 'recording.npz'
                obj_file = dir / 'mesh.obj'
                if npz_file.exists() and obj_file.exists():
                    self.data_files.append((obj_file, npz_file))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        obj_path, npz_path = self.data_files[idx]
        
        # Load mesh file
        mesh = trimesh.load(obj_path)
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        # Load npz data
        npz_data = np.load(npz_path)
        # Convert numpy arrays to torch tensors
        npz_tensors = {k: torch.from_numpy(v) for k, v in npz_data.items()}
        
        # Combine both data sources
        return {
            'vertices': vertices,
            'faces': faces,
            **npz_tensors
        }
