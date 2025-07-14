import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import zipfile

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

        # MUCH larger cache for better performance
        self._scene_cache = {}
        self._cache_size_limit = 1000  # Increased from 100
        self._sdf_cache = {}  # Separate SDF cache
        self._sdf_cache_limit = 500

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
        
        try:
            with np.load(self.data_files[scene_idx]) as scene_data:
                scene = {
                    'sdf': torch.from_numpy(scene_data["sdf"]).float(),
                    'grasps': torch.from_numpy(scene_data["grasps"]).float(),
                    'scores': torch.from_numpy(scene_data["scores"]).float()
                }
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            print(f"Warning: Corrupted data file {self.data_files[scene_idx]}: {e}")
            # Find a valid replacement scene
            for fallback_idx in range(len(self.data_files)):
                if fallback_idx != scene_idx:
                    try:
                        with np.load(self.data_files[fallback_idx]) as fallback_data:
                            scene = {
                                'sdf': torch.from_numpy(fallback_data["sdf"]).float(),
                                'grasps': torch.from_numpy(fallback_data["grasps"]).float(),
                                'scores': torch.from_numpy(fallback_data["scores"]).float()
                            }
                            print(f"Using fallback scene {fallback_idx} instead of {scene_idx}")
                            break
                    except (zipfile.BadZipFile, OSError, ValueError):
                        continue
            else:
                raise RuntimeError(f"Could not find any valid scene data files")
        
        self._scene_cache[scene_idx] = scene
        return scene

    def get_sdf(self, scene_idx):
        """Get SDF with enhanced caching."""
        if scene_idx in self._sdf_cache:
            return self._sdf_cache[scene_idx]
        
        # Load SDF
        scene = self._load_scene(scene_idx)
        sdf = scene['sdf']
        
        # Cache SDF separately
        if len(self._sdf_cache) >= self._sdf_cache_limit:
            # Remove oldest half
            old_keys = list(self._sdf_cache.keys())[:len(self._sdf_cache)//2]
            for key in old_keys:
                del self._sdf_cache[key]
        
        self._sdf_cache[scene_idx] = sdf
        return sdf

    def batch_get_sdf(self, scene_indices):
        """
        Efficiently load multiple SDFs at once.
        Args:
            scene_indices: List or tensor of scene indices
        Returns:
            torch.Tensor: Stacked SDFs of shape (batch_size, 48, 48, 48)
        """
        if isinstance(scene_indices, torch.Tensor):
            scene_indices = scene_indices.cpu().numpy()
        
        sdf_list = []
        uncached_indices = []
        cached_sdfs = {}
        
        # First, collect all cached SDFs
        for i, scene_idx in enumerate(scene_indices):
            scene_idx = int(scene_idx)
            if scene_idx in self._sdf_cache:
                cached_sdfs[i] = self._sdf_cache[scene_idx]
            else:
                uncached_indices.append((i, scene_idx))
        
        # Batch load uncached SDFs
        for i, scene_idx in uncached_indices:
            scene = self._load_scene(scene_idx)
            sdf = scene['sdf']
            
            # Cache the SDF
            if len(self._sdf_cache) >= self._sdf_cache_limit:
                # Remove oldest half
                old_keys = list(self._sdf_cache.keys())[:len(self._sdf_cache)//2]
                for key in old_keys:
                    del self._sdf_cache[key]
            
            self._sdf_cache[scene_idx] = sdf
            cached_sdfs[i] = sdf
        
        # Reconstruct the batch in correct order
        sdf_list = [cached_sdfs[i] for i in range(len(scene_indices))]
        return torch.stack(sdf_list)

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
                

class PreEncodedGraspDataset(Dataset):
    """
    Ultra-fast dataset that pre-encodes all unique SDFs once per epoch.
    This eliminates the 87.7% SDF loading bottleneck by:
    1. Loading all unique SDFs once per epoch
    2. Batch encoding them on GPU
    3. Storing encoded features in GPU memory for instant lookup
    """
    def __init__(self, data_path, model, device, shuffle_grasps=True):
        super().__init__()
        self.data_path = data_path
        self.model = model
        self.device = device
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() if dir.is_dir() and (dir / 'scene.npz').exists()]
        self.shuffle_grasps = shuffle_grasps

        # Create list of (scene_idx, grasp_idx) tuples
        self.grasp_locations = [(scene_idx, grasp_idx) for scene_idx in range(len(self.data_files)) for grasp_idx in range(480)]

        if self.shuffle_grasps:
            self.grasp_indices = torch.randperm(len(self.grasp_locations))
        else:
            self.grasp_indices = list(range(len(self.grasp_locations)))

        # Pre-encoded SDF features cache (GPU memory)
        self.sdf_features_cache = {}
        self.grasp_score_cache = {}
        
        print(f"PreEncodedGraspDataset initialized with {len(self.data_files)} unique scenes")
    
    def pre_encode_epoch(self, scene_indices_needed):
        """
        Pre-encode all unique SDFs needed for this epoch.
        This is called once per epoch before training starts.
        """
        unique_scene_indices = list(set(scene_indices_needed))
        print(f"Pre-encoding {len(unique_scene_indices)} unique SDFs for this epoch...")
        
        # Clear previous cache
        self.sdf_features_cache.clear()
        self.grasp_score_cache.clear()
        
        # Batch encode SDFs for maximum GPU utilization
        batch_size = 32  # Encode 32 SDFs at a time
        
        with torch.no_grad():
            for i in range(0, len(unique_scene_indices), batch_size):
                batch_scene_indices = unique_scene_indices[i:i+batch_size]
                
                # Load SDFs for this batch
                sdf_batch = []
                valid_indices = []
                
                for scene_idx in batch_scene_indices:
                    try:
                        sdf, grasps, scores = self._load_scene_data(scene_idx)
                        sdf_batch.append(sdf)
                        valid_indices.append(scene_idx)
                        
                        # Cache grasp-score pairs
                        self.grasp_score_cache[scene_idx] = (grasps, scores)
                        
                    except Exception as e:
                        print(f"Skipping corrupted scene {scene_idx}: {e}")
                        continue
                
                if sdf_batch:
                    # Stack and move to GPU
                    sdf_tensor = torch.stack(sdf_batch).to(self.device)
                    
                    # Batch encode on GPU
                    encoded_features = self.model.encode_sdf(sdf_tensor)
                    
                    # Store in cache
                    for j, scene_idx in enumerate(valid_indices):
                        self.sdf_features_cache[scene_idx] = encoded_features[j]
                
                # Progress update
                if (i // batch_size) % 10 == 0:
                    print(f"Pre-encoded {min(i + batch_size, len(unique_scene_indices))}/{len(unique_scene_indices)} scenes...")
        
        print(f"Pre-encoding complete! Cached {len(self.sdf_features_cache)} SDF features on GPU")
    
    def _load_scene_data(self, scene_idx):
        """Load raw scene data from disk."""
        try:
            with np.load(self.data_files[scene_idx]) as scene_data:
                sdf = torch.from_numpy(scene_data["sdf"]).float()
                grasps = torch.from_numpy(scene_data["grasps"]).float()
                scores = torch.from_numpy(scene_data["scores"]).float()
                return sdf, grasps, scores
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            # Find a valid replacement
            for fallback_idx in range(len(self.data_files)):
                if fallback_idx != scene_idx:
                    try:
                        with np.load(self.data_files[fallback_idx]) as fallback_data:
                            sdf = torch.from_numpy(fallback_data["sdf"]).float()
                            grasps = torch.from_numpy(fallback_data["grasps"]).float()
                            scores = torch.from_numpy(fallback_data["scores"]).float()
                            return sdf, grasps, scores
                    except (zipfile.BadZipFile, OSError, ValueError):
                        continue
            raise RuntimeError(f"Could not find any valid scene data files")

    def __len__(self):
        return len(self.grasp_locations)

    def __getitem__(self, idx):
        scene_idx, grasp_idx = self.grasp_locations[idx]
        
        # Get pre-encoded SDF features (instant lookup from GPU memory)
        if scene_idx not in self.sdf_features_cache:
            raise RuntimeError(f"Scene {scene_idx} not pre-encoded! Call pre_encode_epoch() first.")
        
        sdf_features = self.sdf_features_cache[scene_idx]
        
        # Get grasp-score data
        grasps, scores = self.grasp_score_cache[scene_idx]
        
        # Handle IndexError by selecting a random valid grasp
        num_grasps = grasps.shape[0]
        if grasp_idx >= num_grasps:
            grasp_idx = np.random.randint(num_grasps)

        return {
            "sdf_features": sdf_features,  # Pre-encoded features (GPU memory)
            "grasp": grasps[grasp_idx],
            "score": scores[grasp_idx],
            "scene_idx": scene_idx,
            "grasp_idx": grasp_idx
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                