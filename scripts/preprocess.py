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
from mesh2sdf import compute
from tqdm import tqdm
from pathlib import Path

# ───────────────────────────────── SDF ────────────────────────────────────
def mesh_to_sdf(mesh:trimesh.Trimesh, n:int, fix_mesh:bool = True) -> np.ndarray:
    """Return (n,n,n) float32 SDF in canonical cube [-1,1]^3 using mesh2sdf."""
    mesh_copy = mesh.copy()
    mesh_copy.apply_translation(-mesh_copy.centroid)
    mesh_copy.apply_scale(1.0 / mesh_copy.extents.max())     # largest dimension == 1, vertices in [-1,1]

    # Call mesh2sdf.compute
    sdf_data = compute(
        vertices=mesh_copy.vertices,
        faces=mesh_copy.faces,
        size=n,
        fix=fix_mesh,
        level=2.0 / n,  # Recommended default: 2/size
        return_mesh=False
    )
    return sdf_data.astype(np.float32)

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
            fix_mesh: bool):
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

        sdf  = mesh_to_sdf(mesh, grid_res, fix_mesh=fix_mesh)

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
    parser.add_argument('-g', '--grid_res', type=int, default=48, help="Grid resolution for voxelization (default: 48).")
    parser.add_argument('-c', '--cores', type=int, default=24, help="Number of CPU cores for parallel processing (default: 24).")
    parser.add_argument('-l', '--limit', type=int, default=None, help="Maximum number of items to process (default: process all).")
    parser.add_argument('--fix_mesh', action='store_true', default=True, help="Attempt to fix non-watertight meshes when computing SDF with mesh2sdf (default: True).")
    parser.add_argument('--no-fix_mesh', dest='fix_mesh', action='store_false', help="Disable fixing meshes for mesh2sdf.")
    args = parser.parse_args()

    GRID_RES = args.grid_res
    CORES = args.cores
    RAW_DIR = Path(args.raw_dir)
    OUT_DIR = Path(args.output_dir)

    t0 = time.time()
    item_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if args.limit is not None and args.limit > 0:
        item_dirs = item_dirs[:args.limit]
        print(f"Found {len(item_dirs):,} items (subdirectories) to process in {RAW_DIR} (limited to first {args.limit})")
    else:
        print(f"Found {len(item_dirs):,} items (subdirectories) to process in {RAW_DIR}")

    worker_fn = functools.partial(process,
                                  grid_res=GRID_RES,
                                  raw_dir=RAW_DIR,
                                  output_dir=OUT_DIR,
                                  fix_mesh=args.fix_mesh)

    print(f"Running in multi-core mode with {CORES} cores.")
    with mp.Pool(CORES, maxtasksperchild=1000) as pool: # prevent resource accumulation by restarting workers after 1000 tasks
        list(tqdm(pool.imap_unordered(worker_fn, item_dirs, chunksize=8), total=len(item_dirs))) # process in chunks of 8

    print(f"All done in {(time.time()-t0)/60:.1f} min → {OUT_DIR}")
