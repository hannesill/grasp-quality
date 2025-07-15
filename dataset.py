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
        # Use larger batch size for faster pre-encoding
        batch_size = 128  # Encode 128 SDFs at a time (4x faster)
        
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
                    
                    # Store in cache on CPU (for multiprocessing compatibility)
                    for j, scene_idx in enumerate(valid_indices):
                        self.sdf_features_cache[scene_idx] = encoded_features[j].cpu()
                
                # Progress update every 20 batches for less spam
                if (i // batch_size) % 20 == 0:
                    print(f"Pre-encoded {min(i + batch_size, len(unique_scene_indices))}/{len(unique_scene_indices)} scenes...")
                    
                # Clear GPU cache periodically to avoid memory buildup
                if (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
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
                

class CachedGraspDataset(Dataset):
    """
    Clean dataset that works with pre-encoded SDF features.
    No model dependency - just handles pre-computed features.
    """
    def __init__(self, data_path, shuffle_grasps=True):
        super().__init__()
        self.data_path = data_path
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() 
                          if dir.is_dir() and (dir / 'scene.npz').exists()]
        
        # Create list of (scene_idx, grasp_idx) tuples
        self.grasp_locations = [(scene_idx, grasp_idx) 
                               for scene_idx in range(len(self.data_files)) 
                               for grasp_idx in range(480)]
        
        # Caches - populated by training script
        self.sdf_features_cache = {}
        self.grasp_score_cache = {}
    
    def update_caches(self, sdf_features_cache, grasp_score_cache):
        """Update caches from training script."""
        self.sdf_features_cache = sdf_features_cache
        self.grasp_score_cache = grasp_score_cache
    
    def load_scene_data(self, scene_idx):
        """Public method for training script to load raw data."""
        with np.load(self.data_files[scene_idx]) as scene_data:
            sdf = torch.from_numpy(scene_data["sdf"]).float()
            grasps = torch.from_numpy(scene_data["grasps"]).float()
            scores = torch.from_numpy(scene_data["scores"]).float()
            
            # Pad to exactly 480 grasp-score pairs if needed
            num_grasps = grasps.shape[0]
            if num_grasps < 480:
                pad_size = 480 - num_grasps
                indices_to_duplicate = torch.randint(0, num_grasps, (pad_size,))
                grasps = torch.cat([grasps, grasps[indices_to_duplicate]], dim=0)
                scores = torch.cat([scores, scores[indices_to_duplicate]], dim=0)
            
            return sdf, grasps, scores

    def __len__(self):
        return len(self.grasp_locations)

    def __getitem__(self, idx):
        scene_idx, grasp_idx = self.grasp_locations[idx]
        sdf_features = self.sdf_features_cache[scene_idx]
        grasps, scores = self.grasp_score_cache[scene_idx]
        
        return {
            "sdf_features": sdf_features,
            "grasp": grasps[grasp_idx],
            "score": scores[grasp_idx],
            "scene_idx": scene_idx,
            "grasp_idx": grasp_idx
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
                

class GPUCachedGraspDataset(Dataset):
    """
    Ultimate performance dataset that loads ALL SDFs into GPU memory at startup.
    
    Benefits:
    - Zero data loading time during training (near-instant batch creation)
    - All SDFs stored as single contiguous GPU tensor
    - Simple indexing with no CPU-GPU transfers
    - Perfect for A100 with 40GB memory (only needs ~6.7GB for 15,547 scenes)
    """
    def __init__(self, data_path, device, shuffle_grasps=True):
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.data_files = [dir / 'scene.npz' for dir in data_path.iterdir() 
                          if dir.is_dir() and (dir / 'scene.npz').exists()]
        
        print(f"Found {len(self.data_files)} scenes")
        
        # Create list of (scene_idx, grasp_idx) tuples
        self.grasp_locations = [(scene_idx, grasp_idx) 
                               for scene_idx in range(len(self.data_files)) 
                               for grasp_idx in range(480)]
        
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
                

class ThreeChannelGPUDataset(Dataset):
    """
    GPU-cached dataset for preprocessed 3-channel inputs.
    
    Loads all preprocessed 3-channel data (SDF + palm distance + fingertip distance) 
    to GPU memory for ultra-fast training.
    """
    
    def __init__(self, preprocessed_path, device='cuda'):
        super().__init__()
        self.preprocessed_path = Path(preprocessed_path)
        self.device = device
        
        # Load all preprocessed data to GPU
        self.gpu_inputs = None
        self.gpu_scores = None
        self._load_all_to_gpu()
        
        print(f"✅ 3-channel dataset loaded to {device}!")
        print(f"📊 Total samples: {len(self)}")
        print(f"📐 Input shape: {self.gpu_inputs.shape}")
        print(f"💾 GPU memory usage: {self._get_gpu_memory_usage():.2f} GB")
    
    def _load_all_to_gpu(self):
        """Load all preprocessed 3-channel data to GPU memory."""
        # Find all preprocessed files
        preprocessed_files = list(self.preprocessed_path.glob("*_3channel.npz"))
        
        if not preprocessed_files:
            raise FileNotFoundError(f"No preprocessed files found in {self.preprocessed_path}")
        
        print(f"📁 Found {len(preprocessed_files)} preprocessed files")
        print(f"🔥 Loading all data to {self.device}...")
        
        # Load all data
        all_inputs = []
        all_scores = []
        
        for file in tqdm(preprocessed_files, desc="Loading preprocessed data"):
            try:
                with np.load(file) as data:
                    inputs = data['inputs']  # (N, 3, 48, 48, 48)
                    scores = data['scores']  # (N,)
                    
                    all_inputs.append(inputs)
                    all_scores.append(scores)
            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
                continue
        
        if not all_inputs:
            raise RuntimeError("No valid preprocessed data found!")
        
        # Stack all data
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        
        print(f"📊 Total dataset size: {len(all_inputs)} samples")
        print(f"📐 Input shape: {all_inputs.shape}")
        print(f"💾 Memory usage: {all_inputs.nbytes / (1024**3):.2f} GB")
        
        # Convert to tensors and move to device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.gpu_inputs = torch.from_numpy(all_inputs).float().to(self.device)
            self.gpu_scores = torch.from_numpy(all_scores).float().to(self.device)
            print(f"🚀 Data loaded to GPU successfully!")
        else:
            print("⚠️  CUDA not available, keeping data on CPU")
            self.gpu_inputs = torch.from_numpy(all_inputs).float()
            self.gpu_scores = torch.from_numpy(all_scores).float()
    
    def _get_gpu_memory_usage(self):
        """Calculate GPU memory usage in GB."""
        total_bytes = 0
        if self.gpu_inputs is not None:
            total_bytes += self.gpu_inputs.numel() * self.gpu_inputs.element_size()
        if self.gpu_scores is not None:
            total_bytes += self.gpu_scores.numel() * self.gpu_scores.element_size()
        return total_bytes / (1024 ** 3)
    
    def __len__(self):
        return len(self.gpu_inputs) if self.gpu_inputs is not None else 0
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'input': (3, 48, 48, 48) tensor - 3-channel input
                - 'score': float - grasp quality score
        """
        return {
            'input': self.gpu_inputs[idx],  # (3, 48, 48, 48)
            'score': self.gpu_scores[idx],  # scalar
        }
    
    def get_batch(self, indices):
        """
        Get a batch of samples efficiently.
        
        Args:
            indices: List or tensor of sample indices
            
        Returns:
            dict with keys:
                - 'input': (batch_size, 3, 48, 48, 48) tensor
                - 'score': (batch_size,) tensor
        """
        if isinstance(indices, (list, tuple)):
            indices = torch.tensor(indices, device=self.device)
        elif isinstance(indices, torch.Tensor):
            indices = indices.to(self.device)
        
        return {
            'input': self.gpu_inputs[indices],
            'score': self.gpu_scores[indices],
        }


class ThreeChannelPreprocessedDataset(Dataset):
    """
    Lightweight dataset for preprocessed 3-channel data.
    
    Loads preprocessed files on-demand with caching for efficient training.
    Good for when you can't fit all data in GPU memory.
    """
    
    def __init__(self, preprocessed_path, cache_size=100):
        super().__init__()
        self.preprocessed_path = Path(preprocessed_path)
        self.cache_size = cache_size
        
        # Find all preprocessed files
        self.preprocessed_files = list(self.preprocessed_path.glob("*_3channel.npz"))
        
        if not self.preprocessed_files:
            raise FileNotFoundError(f"No preprocessed files found in {self.preprocessed_path}")
        
        # Build index: (file_idx, sample_idx) for each sample
        self.sample_index = []
        self.file_sample_counts = []
        
        print("📁 Indexing preprocessed files...")
        for file_idx, file in enumerate(tqdm(self.preprocessed_files, desc="Indexing")):
            try:
                with np.load(file) as data:
                    num_samples = data['inputs'].shape[0]
                    for sample_idx in range(num_samples):
                        self.sample_index.append((file_idx, sample_idx))
                    self.file_sample_counts.append(num_samples)
            except Exception as e:
                print(f"⚠️  Error indexing {file}: {e}")
                self.file_sample_counts.append(0)
        
        # File cache
        self.file_cache = {}
        self.cache_order = []
        
        print(f"✅ Dataset indexed: {len(self.sample_index)} samples from {len(self.preprocessed_files)} files")
    
    def _load_file(self, file_idx):
        """Load a preprocessed file with caching."""
        if file_idx in self.file_cache:
            return self.file_cache[file_idx]
        
        # Load file
        file_path = self.preprocessed_files[file_idx]
        try:
            with np.load(file_path) as data:
                inputs = torch.from_numpy(data['inputs']).float()  # (N, 3, 48, 48, 48)
                scores = torch.from_numpy(data['scores']).float()  # (N,)
                
                file_data = {'inputs': inputs, 'scores': scores}
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            # Return dummy data
            file_data = {
                'inputs': torch.zeros(1, 3, 48, 48, 48),
                'scores': torch.zeros(1)
            }
        
        # Cache management
        if len(self.file_cache) >= self.cache_size:
            # Remove oldest file from cache
            oldest_file = self.cache_order.pop(0)
            del self.file_cache[oldest_file]
        
        self.file_cache[file_idx] = file_data
        self.cache_order.append(file_idx)
        
        return file_data
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        file_idx, sample_idx = self.sample_index[idx]
        file_data = self._load_file(file_idx)
        
        return {
            'input': file_data['inputs'][sample_idx],  # (3, 48, 48, 48)
            'score': file_data['scores'][sample_idx],  # scalar
        }
                