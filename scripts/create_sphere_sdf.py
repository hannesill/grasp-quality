import numpy as np
import argparse
import os

def create_spherical_sdf(size):
    """
    Creates a spherical SDF where the value is the distance from the center.
    The minimum (0) is at the center of the volume.
    """
    coords = np.linspace(-1.0, 1.0, size)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    
    # Distance from center (0,0,0)
    sdf = np.sqrt(x**2 + y**2 + z**2)
    
    return sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an artificial spherical SDF file.")
    parser.add_argument('-s', '--size', type=int, default=48, help="Size of the SDF grid (default: 48).")
    parser.add_argument('-o', '--output', type=str, default='data/sphere.npz', help="Output file path for the .npz file (default: 'data/sphere.npz').")
    parser.add_argument('-k', '--key', type=str, default='sdf', help="Key to save the SDF under in the .npz file (default: 'sdf')")
    args = parser.parse_args()

    print(f"Generating spherical SDF of size {args.size}x{args.size}x{args.size}...")
    sdf_data = create_spherical_sdf(args.size)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(args.output, **{args.key: sdf_data})
    print(f"SDF saved to {args.output}")
