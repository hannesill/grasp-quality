import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import trimesh
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def visualize_sdf(sdf, output_dir):
    mesh_scale = 0.8
    levels = [-0.02, 0.0, 0.02]

    size = sdf.shape[0]
    sdf_min, sdf_max = sdf.min(), sdf.max()
    print(sdf_max, sdf_min)

    # extract level sets
    for i, level in enumerate(levels):
        if not (sdf_min < level < sdf_max):
            continue
        vtx, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

        vtx = vtx * (mesh_scale * 2.0 / size) - 1.0
        mesh = trimesh.Trimesh(vtx, faces)
        mesh.export(os.path.join(output_dir, 'l%.2f.obj' % level))


    # draw image
    for i in range(size):
        array_2d = sdf[:, :, i]

        num_levels = 6
        fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))

        sample = array_2d
        # sample = np.flipud(array_2d)
        CS = ax.contourf(sample, levels=levels, colors=colors)

        ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
        ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
        ax.axis('off')

        plt.savefig(os.path.join(output_dir, '%03d.png' % i))
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview SDF and save level sets and slices.")
    parser.add_argument('-n', '--number', type=str, default='0', help="Experiment number to preview (default: '0').")
    parser.add_argument('-f', '--file', type=str, help="Alternative: direct path to scene.npz file")
    parser.add_argument('-k', '--key', type=str, default='sdf', help="Key to read from the .npz file (default: 'sdf')")
    parser.add_argument('-o', '--output', type=str, default='data/preview/', help="Output directory for level sets and slices")
    args = parser.parse_args()

    if args.file:
        filename = args.file
        if filename.startswith('data/processed'):
            args.number = filename.split('/')[2]
            output_dir = os.path.join(args.output, args.number)
        else:
            output_dir = args.output + '/' + filename.split('/')[-1].split('.')[0]
    else:
        filename = os.path.join(os.path.dirname(__file__), 'data', 'processed', args.number, 'scene.npz')
        output_dir = os.path.join(args.output, args.number)

    
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(filename)
    if args.key not in data:
        raise ValueError(f"Key '{args.key}' not found in the .npz file")

    visualize_sdf(data[args.key], output_dir)