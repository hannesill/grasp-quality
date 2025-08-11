"""
Pre-compute n³ SDFs + 19-ch grasp volumes for a whole dataset.

Usage
-----
python preprocess.py --raw_dir /data/raw --output_dir /data/processed --grid_res 48 --cores 24
"""
import functools, multiprocessing as mp, time, argparse
import numpy as np
import trimesh
from mesh2sdf import compute
from tqdm import tqdm
from pathlib import Path
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ──────────────────────────── Mesh Max Extent ────────────────────────────
def get_mesh_max_extent(item_dirs: list[Path]) -> float:
    max_extent = 0
    for item_dir in tqdm(item_dirs, desc="Analyzing meshes for scaling"):
        obj_path = item_dir / 'mesh.obj'
        if not obj_path.exists():
            continue

        try:
            mesh = trimesh.load(str(obj_path), force='mesh')
            if not mesh.vertices.size or not mesh.faces.size:
                continue
            max_extent = max(max_extent, mesh.extents.max())
        except Exception as e:
            print(f"Warning: could not load mesh {obj_path} to determine scale: {e}")

    if max_extent == 0:
        raise RuntimeError("Could not determine maximum extent from any mesh.")

    return max_extent

# ───────────────────────────────── SDF ────────────────────────────────────
def mesh_to_sdf(mesh: trimesh.Trimesh,
                n: int,
                fix_mesh: bool = False,
                global_scale: float = None) -> np.ndarray:
    """Return (n,n,n) float32 SDF in canonical cube [-1,1]^3 using mesh2sdf."""
    mesh_copy = mesh.copy()
    translation = mesh_copy.bounds.mean(axis=0)  # center the mesh
    mesh_copy.apply_translation(-translation)
    mesh_copy.apply_scale(global_scale) # rescale uniformly so all meshes are in [-1,1]^3

    sdf_data = compute(vertices=mesh_copy.vertices,
                       faces=mesh_copy.faces,
                       size=n,
                       fix=fix_mesh,
                       level=2.0 / n,  # Recommended default: 2/size
                       return_mesh=False)
    return sdf_data.astype(np.float32), translation

# ─────────────────────────────── Per-file job ────────────────────────────
def process(item_dir: Path, grid_res: int, raw_dir: Path, output_dir: Path,
            fix_mesh: bool, global_scale: float):
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

        sdf, translation = mesh_to_sdf(mesh, grid_res, fix_mesh=fix_mesh, global_scale=global_scale)

        recording = np.load(str(npz_path))

        if 'grasps' not in recording or recording['grasps'].shape[0] == 0:
            print(f"Warning: No grasps found or grasps array is empty in {npz_path}. Skipping {item_dir}.")
            return
        if 'scores' not in recording:
            print(f"Warning: 'scores' not found in {npz_path}. Skipping {item_dir}.")
            return

        grasps = recording['grasps']
        # Apply translation and scaling to grasp positions
        grasps[:, :3] = (grasps[:, :3] - translation) * global_scale

        np.savez_compressed(str(out_path),
                            sdf=sdf,
                            grasps=grasps,
                            scores=recording['scores'],
                            translation=translation,
                            global_scale=global_scale)

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
    parser.add_argument('--fix_mesh', action='store_true', default=False, help="Attempt to fix non-watertight meshes when computing SDF with mesh2sdf (default: False).")
    parser.add_argument('--mesh_max_extent', type=float, default=None, help="Pre-computed mesh max extent. If provided, skips mesh analysis for scaling.")
    args = parser.parse_args()

    GRID_RES = args.grid_res
    CORES = args.cores
    RAW_DIR = Path(args.raw_dir)
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    item_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if args.limit is not None and args.limit > 0:
        item_dirs = item_dirs[:args.limit]
        print(f"Found {len(item_dirs):,} items (subdirectories) to process in {RAW_DIR} (limited to first {args.limit})")
    else:
        print(f"Found {len(item_dirs):,} items (subdirectories) to process in {RAW_DIR}")

    # ────────────────────────────────── Mesh Scale ─────────────────────────
    if args.mesh_max_extent:
        mesh_max_extent = args.mesh_max_extent
        print(f"Using provided mesh max extent: {mesh_max_extent}")
    else:
        mesh_max_extent = get_mesh_max_extent(item_dirs)
        print(f"Mesh max extent: {mesh_max_extent}")

    # Save the scale factor for later use
    mesh_max_extent_path = OUT_DIR / 'mesh_max_extent.txt'
    with mesh_max_extent_path.open('w') as f:
        f.write(str(mesh_max_extent))
    print(f"Saved mesh max extent to {mesh_max_extent_path}")

    global_scale = 2.0 / mesh_max_extent # rescale uniformly so all meshes are in [-1,1]^3
    print(f"Global uniform scale factor: {global_scale}")

    worker_fn = functools.partial(process,
                                  grid_res=GRID_RES,
                                  raw_dir=RAW_DIR,
                                  output_dir=OUT_DIR,
                                  fix_mesh=args.fix_mesh,
                                  global_scale=global_scale)

    print(f"Running in multi-core mode with {CORES} cores.")
    with mp.Pool(CORES, maxtasksperchild=1000) as pool:  # prevent resource accumulation by restarting workers after 1000 tasks
        list(tqdm(pool.imap_unordered(worker_fn, item_dirs, chunksize=8), total=len(item_dirs)))  # process in chunks of 8

    print(f"All done in {(time.time()-t0)/60:.1f} min → {OUT_DIR}")
