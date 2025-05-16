"""
Pre-compute n³ SDFs + 19-ch grasp volumes for a whole dataset.

Usage
-----
python preprocess.py --raw_dir /data/raw --output_dir /data/processed --grid_res 48 --cores 24
"""
import functools, multiprocessing as mp, time, argparse
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
from tqdm import tqdm
from pathlib import Path

try:
    import torch
    import torchcumesh2sdf
    TORCHCUMESH2SDF_AVAILABLE = True
except ImportError:
    TORCHCUMESH2SDF_AVAILABLE = False

# ───────────────────────────────── SDF ────────────────────────────────────
def mesh_to_sdf(mesh:trimesh.Trimesh, n:int, use_gpu:bool, gpu_sdf_band_factor:float, sign_method:str='normal', scan_count:int=100, scan_resolution:int=400) -> np.ndarray:
    """Return (n,n,n) float32 SDF in canonical cube [-1,1]^3."""
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.extents.max())     # largest dimension == 1

    if use_gpu and TORCHCUMESH2SDF_AVAILABLE:
        if not (n > 0 and (n & (n - 1) == 0)): # Check if n is power of 2
            print(f"Warning: Grid resolution {n} is not a power of 2. GPU SDF computation via cumesh2sdf requires power-of-2 resolution. Falling back to CPU for this item.")
        else:
            try:
                if not torch.cuda.is_available():
                    print("Warning: CUDA not available for PyTorch. Falling back to CPU for SDF computation.")
                else:
                    tris_tensor = torch.tensor(mesh.triangles, dtype=torch.float32, device='cuda')
                    band = float(gpu_sdf_band_factor) / n
                    # Assuming mesh.triangles are already in the normalized space due to pre-processing
                    sdf_tensor = torchcumesh2sdf.get_sdf(tris_tensor, R=n, band=band)
                    return sdf_tensor.cpu().numpy()
            except Exception as e:
                print(f"Error during GPU SDF computation: {e}. Falling back to CPU.")
    
    # Fallback to CPU method (original mesh_to_voxels)
    return mesh_to_voxels(mesh, voxel_resolution=n, sign_method=sign_method, scan_count=scan_count, scan_resolution=scan_resolution).astype(np.float32)

# ────────────────────────────── Grasps → volume ──────────────────────────
def grasps_to_volume(g:np.ndarray, n:int, sigma:float=1.5) -> np.ndarray:
    """
    g : (B,19)  – [x,y,z, qx,qy,qz,qw, open width, …]  range ≈ [-1,1]
    Returns (B,n,n,n) float32 with a 3-D Gaussian splat per grasp, summed across all 19 channels.
    """
    B = len(g)
    vol = np.zeros((B, n, n, n), dtype=np.float32)
    
    grid = (np.arange(n) + 0.5) / n * 2 - 1          # (n,)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')  # (n,n,n)
    
    centers = g[:, :3][:, None, None, None, :]  # (B,1,1,1,3)
    coords = np.stack([X,Y,Z], axis=-1)         # (n,n,n,3)
    
    dd = np.sum((coords - centers)**2, axis=-1) # (B,n,n,n)
    masks = np.exp(-dd / (2*sigma**2))          # (B,n,n,n)
    
    g_expanded = g[:, None, None, None, :]      # (B,1,1,1,19)
    vol = np.sum(masks[..., None] * g_expanded, axis=-1)  # (B,n,n,n)
    
    return vol.astype(np.float32)

# ─────────────────────────────── Per-file job ────────────────────────────
def process(item_dir: Path, grid_res: int, raw_dir: Path, output_dir: Path, 
            sign_method:str, scan_count:int, scan_resolution:int,
            use_gpu:bool, gpu_sdf_band_factor:float):
    obj_path = item_dir / 'mesh.obj'
    npz_path = item_dir / 'recording.npz'

    # Output path should maintain the relative structure from raw_dir
    item_relative_path = item_dir.relative_to(raw_dir)
    out_path = output_dir / item_relative_path / 'scene.npz'

    if out_path.exists():
        return

    if not obj_path.exists():
        print(f"Warning: Mesh file .obj not found in {item_dir} (expected at {obj_path}). Skipping.")
        return

    if not npz_path.exists():
        print(f"Warning: Grasp file recording.npz not found in {item_dir} (expected at {npz_path}). Skipping.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh = trimesh.load(str(obj_path), force='mesh')
        if not mesh.vertices.size or not mesh.faces.size:
            print(f"Warning: Mesh loaded from {obj_path} is empty or invalid. Skipping {item_dir}.")
            return

        sdf  = mesh_to_sdf(mesh, grid_res, use_gpu=use_gpu, gpu_sdf_band_factor=gpu_sdf_band_factor, 
                           sign_method=sign_method, scan_count=scan_count, scan_resolution=scan_resolution)

        recording = np.load(str(npz_path))
        
        if 'grasps' not in recording or recording['grasps'].shape[0] == 0:
            print(f"Warning: No grasps found or grasps array is empty in {npz_path}. Skipping {item_dir}.")
            return
        if 'scores' not in recording:
            print(f"Warning: 'scores' not found in {npz_path}. Skipping {item_dir}.")
            return
    
        grasps = recording['grasps']
        # gvol = grasps_to_volume(grasps, grid_res)

        np.savez_compressed(str(out_path), sdf=sdf, grasps=grasps, scores=recording['scores'])

    except Exception as e:
        print(f"Error processing {item_dir}: {e}")

# ────────────────────────────────── Main ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute SDFs and grasp volumes for a dataset.")
    parser.add_argument('-r', '--raw_dir', type=str, default='data/raw', help="Input directory containing raw mesh files (e.g., /data/raw_meshes).")
    parser.add_argument('-o', '--output_dir', type=str, default='data/processed', help="Output directory to save precomputed SDFs and grasp volumes (e.g., /data/processed_data).")
    parser.add_argument('-g', '--grid_res', type=int, default=48, help="Grid resolution for voxelization (default: 48). For GPU mode, this must be a power of 2 (e.g., 32, 64).")
    parser.add_argument('-c', '--cores', type=int, default=24, help="Number of CPU cores for parallel processing (default: 24).")
    parser.add_argument('-l', '--limit', type=int, default=None, help="Maximum number of items to process (default: process all).")
    parser.add_argument('--sign_method', type=str, default='normal', choices=['normal', 'depth'], help="CPU SDF: Method for determining SDF sign (default: normal). 'depth' is faster but requires watertight meshes.")
    parser.add_argument('--scan_count', type=int, default=100, help="CPU SDF: Number of scans for SDF generation (default: 100).")
    parser.add_argument('--scan_resolution', type=int, default=400, help="CPU SDF: Resolution of scans for SDF generation (default: 400).")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for SDF computation via cumesh2sdf (if available). Requires --grid_res to be a power of 2.")
    parser.add_argument('--gpu_sdf_band_factor', type=float, default=3.0, help="GPU SDF: Factor to determine band size (band = factor / grid_res). Default: 3.0.")
    args = parser.parse_args()

    GRID_RES = args.grid_res
    CORES = args.cores
    RAW_DIR = Path(args.raw_dir)
    OUT_DIR = Path(args.output_dir)

    if args.use_gpu and not TORCHCUMESH2SDF_AVAILABLE:
        print("Warning: --use_gpu was specified, but torchcumesh2sdf library is not available. Falling back to CPU.")
    elif args.use_gpu and TORCHCUMESH2SDF_AVAILABLE and not torch.cuda.is_available():
        print("Warning: --use_gpu was specified, but PyTorch CUDA is not available. Falling back to CPU.")
    elif args.use_gpu and TORCHCUMESH2SDF_AVAILABLE and not (args.grid_res > 0 and (args.grid_res & (args.grid_res - 1) == 0)):
        print(f"Warning: --use_gpu was specified, but grid_res ({args.grid_res}) is not a power of 2. GPU SDF computation will fallback to CPU if this is not corrected for each item, or you might encounter errors.")

    t0 = time.time()
    item_dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if args.limit is not None and args.limit > 0:
        item_dirs = item_dirs[:args.limit]
        print(f"Found {len(item_dirs):,} items (subdirectories) to process in {RAW_DIR} (limited to first {args.limit})")
    else:
        print(f"Found {len(item_dirs):,} items (subdirectories) to process in {RAW_DIR}")

    worker_fn = functools.partial(process,
                                  grid_res=GRID_RES,
                                  raw_dir=RAW_DIR,
                                  output_dir=OUT_DIR,
                                  sign_method=args.sign_method,
                                  scan_count=args.scan_count,
                                  scan_resolution=args.scan_resolution,
                                  use_gpu=args.use_gpu,
                                  gpu_sdf_band_factor=args.gpu_sdf_band_factor)

    print(f"Running in multi-core mode with {CORES} cores.")
    with mp.Pool(CORES, maxtasksperchild=1) as pool:
        list(tqdm(pool.imap_unordered(worker_fn, item_dirs, chunksize=1), total=len(item_dirs)))

    print(f"All done in {(time.time()-t0)/60:.1f} min → {OUT_DIR}")
