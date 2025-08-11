import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_sdf_from_args(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """
    Load SDF volume from either a provided file path or a scene number under data/processed.

    Returns
    -------
    sdf : np.ndarray
        The loaded SDF volume as a float array shaped (N, N, N).
    output_dir : str
        Output directory to save generated images.
    """
    if args.file:
        filename = args.file
        if filename.startswith('data/processed'):
            args.number = filename.split('/')[2]
            output_dir = os.path.join(args.output, args.number)
        else:
            output_dir = os.path.join(args.output, os.path.splitext(os.path.basename(filename))[0])
    else:
        filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', args.number, 'scene.npz')
        output_dir = os.path.join(args.output, args.number)

    filename = os.path.abspath(filename)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find input file: {filename}")

    data = np.load(filename)
    if args.key not in data:
        raise KeyError(f"Key '{args.key}' not found in {filename}. Available: {list(data.keys())}")
    sdf = data[args.key]
    if sdf.ndim != 3 or sdf.shape[0] != sdf.shape[1] or sdf.shape[1] != sdf.shape[2]:
        raise ValueError(f"Expected cubic 3D SDF volume, got shape {sdf.shape}")

    os.makedirs(output_dir, exist_ok=True)
    return sdf.astype(np.float32), output_dir


def compute_color_norm(sdf: np.ndarray,
                       symmetric: bool,
                       vmin: float | None,
                       vmax: float | None,
                       percentile: float | None) -> tuple[TwoSlopeNorm, float, float]:
    """
    Create a consistent normalization centered at 0 for all slices.

    Priority: explicit vmin/vmax > percentile > symmetric full-range > data min/max.
    """
    if vmin is not None and vmax is not None:
        lo, hi = float(vmin), float(vmax)
    elif percentile is not None:
        p = float(percentile)
        p = max(0.0, min(100.0, p))
        if symmetric:
            bound = np.percentile(np.abs(sdf), p)
            lo, hi = -bound, bound
        else:
            lo, hi = np.percentile(sdf, [100 - p, p])
    elif symmetric:
        bound = max(abs(float(sdf.min())), abs(float(sdf.max())))
        lo, hi = -bound, bound
    else:
        lo, hi = float(sdf.min()), float(sdf.max())

    # Avoid degenerate range
    if hi <= lo:
        eps = 1e-6
        lo, hi = lo - eps, hi + eps

    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    return norm, lo, hi


def extract_slice(sdf: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == 'x':
        return sdf[index, :, :]
    elif axis == 'y':
        return sdf[:, index, :]
    elif axis == 'z':
        return sdf[:, :, index]
    else:
        raise ValueError(f"Invalid axis '{axis}'. Choose from 'x', 'y', 'z'.")


def save_sdf_slices(
    sdf: np.ndarray,
    output_dir: str,
    axis: str = 'z',
    every: int = 1,
    start: int | None = None,
    end: int | None = None,
    cmap: str = 'Spectral',
    dpi: int = 300,
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile: float | None = None,
    draw_zero_contour: bool = False,
):
    size = sdf.shape[0]

    # Determine global color normalization
    norm, lo, hi = compute_color_norm(sdf, symmetric=symmetric, vmin=vmin, vmax=vmax, percentile=percentile)

    # Determine slice range
    i0 = 0 if start is None else max(0, int(start))
    i1 = size if end is None else min(size, int(end))
    step = max(1, int(every))

    for i in range(i0, i1, step):
        array_2d = extract_slice(sdf, axis=axis, index=i)

        fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=dpi)
        im = ax.imshow(array_2d, cmap=cmap, norm=norm, origin='lower')

        if draw_zero_contour:
            ax.contour(array_2d, levels=[0.0], colors='k', linewidths=0.3)

        ax.set_title(f"{axis.upper()} slice {i}")
        ax.set_xlabel('voxel')
        ax.set_ylabel('voxel')

        # Add colorbar with label
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SDF value')

        out_name = os.path.join(output_dir, f"slice_{axis}_{i:03d}.png")
        fig.tight_layout()
        fig.savefig(out_name)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save per-slice SDF images with a colorbar (value scale).")
    parser.add_argument('-n', '--number', type=str, default='0', help="Experiment number under data/processed to preview (default: '0').")
    parser.add_argument('-f', '--file', type=str, help="Optional: direct path to a .npz file (e.g., data/processed/<id>/scene.npz)")
    parser.add_argument('-k', '--key', type=str, default='sdf', help="Key to read from the .npz file (default: 'sdf')")
    parser.add_argument('-o', '--output', type=str, default='data/preview_slices/', help="Output directory for rendered slices")

    parser.add_argument('--axis', type=str, default='z', choices=['x', 'y', 'z'], help="Axis along which to slice (default: z)")
    parser.add_argument('--every', type=int, default=1, help="Stride when iterating over slices (default: 1)")
    parser.add_argument('--start', type=int, default=None, help="First slice index to render (default: 0)")
    parser.add_argument('--end', type=int, default=None, help="One-past-last slice index to render (default: size)")

    parser.add_argument('--dpi', type=int, default=300, help="Output image DPI (default: 300)")
    parser.add_argument('--cmap', type=str, default='Spectral', help="Matplotlib colormap name (default: 'Spectral')")

    parser.add_argument('--symmetric', action='store_true', default=True, help="Use symmetric color limits around 0 (default: True)")
    parser.add_argument('--no-symmetric', dest='symmetric', action='store_false', help="Disable symmetric color limits around 0")
    parser.add_argument('--vmin', type=float, default=None, help="Explicit lower color limit (overrides percentile)")
    parser.add_argument('--vmax', type=float, default=None, help="Explicit upper color limit (overrides percentile)")
    parser.add_argument('--percentile', type=float, default=None, help="Clip color limits to +/- percentile of values (e.g., 99)")
    parser.add_argument('--zero-contour', action='store_true', help="Overlay a contour line at SDF=0")

    args = parser.parse_args()

    # Ensure project root on path if needed later
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    sdf, output_dir = load_sdf_from_args(args)

    save_sdf_slices(
        sdf=sdf,
        output_dir=output_dir,
        axis=args.axis,
        every=args.every,
        start=args.start,
        end=args.end,
        cmap=args.cmap,
        dpi=args.dpi,
        symmetric=args.symmetric,
        vmin=args.vmin,
        vmax=args.vmax,
        percentile=args.percentile,
        draw_zero_contour=args.zero_contour,
    )


if __name__ == '__main__':
    main()


