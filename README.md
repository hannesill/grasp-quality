# Grasp Quality Estimation

End-to-end pipeline for grasp quality estimation on SDF volumes: preprocessing, training, sampling, optimization, and visualization.

## Environment setup (conda)

This repo targets the `adlr` conda env defined in `environment.yml`.

1) Create/update env
```bash
conda env create -f environment.yml || conda env update -n adlr -f environment.yml --prune
```

2) Activate env (required for all commands below)
```bash
conda activate adlr
```

Notes
- Python 3.10 is expected in this env. Running with system Python may fail.
- If some packages are missing, run: `conda env update -n adlr -f environment.yml --prune`.

## Data layout

Expected directories:
- `data/raw/<scene_id>/{mesh.obj, recording.npz}`: raw mesh and original grasps+scores
- `data/processed/<scene_id>/scene.npz`: produced by preprocessing; contains `sdf, grasps, scores, translation, global_scale`
- `data/splits.json`: train/val/test split ids

## Quick sanity checks

Create a toy SDF and render slices:
```bash
conda activate adlr
python scripts/data_processing/create_sphere_sdf.py -s 32 -o data/sphere.npz
python scripts/visualizations/vis_sdf_slices.py -f data/sphere.npz -o data/preview_slices
```

## Scripts overview and usage

Preprocessing:
- `scripts/data_processing/preprocess.py`
  ```bash
  python scripts/data_processing/preprocess.py \
    -r data/raw \
    -o data/processed \
    -g 48 \
    --fix_mesh \
    --mesh_max_extent 0   # optional; auto-computed if omitted
  ```
  Produces `scene.npz` files and writes `mesh_max_extent.txt` in output.

Splits:
- `scripts/data_processing/create_splits.py`
  ```bash
  python scripts/data_processing/create_splits.py --data_path data/processed --train_ratio 0.8 --val_ratio 0.1 --output_path data/splits.json
  ```

Training GQ model:
- `scripts/training/train.py`
  ```bash
  python scripts/training/train.py \
    --data_path data/processed \
    --splits_file data/splits.json \
    --epochs 100 --batch_size 32 --base_channels 16 --fc_dims 256 128 64
  ```
  Outputs `final_model.pth` under Weights & Biases run dir; best checkpoint saved as `best_model.pth` in cwd when val improves after warmup.

Train object autoencoder (optional):
- `scripts/training/train_autoencoder.py`
  ```bash
  python scripts/training/train_autoencoder.py --data_path data/processed --epochs 100
  ```

Train normalizing flow (grasp prior):
- `scripts/training/train_nflow.py`
  ```bash
  python scripts/training/train_nflow.py --data_path data/processed --epochs 100
  ```

Train diffusion model (optional):
- `scripts/training/train_diff.py`
  ```bash
  python scripts/training/train_diff.py --data_path data/processed --epochs 500 --encoder_path checkpoints/object_encoder.pth
  ```

Evaluation and plots:
- `scripts/testing/eval_gq_model_plot.py` (targets-vs-preds plot)
  ```bash
  python scripts/testing/eval_gq_model_plot.py --model_path final_model.pth --split test --output plots/gq_ordering_final.png
  ```

Sampling grasps:
- `scripts/optimization/sample_grasp.py` (nflow or diffusion)
  ```bash
  # Normalizing flow
  python scripts/optimization/sample_grasp.py --model_type nflow --model_path best_nflow_model.pth --num_samples 50 --output_path data/sampled --scene_name demo
  # Diffusion (needs encoder and sdf)
  python scripts/optimization/sample_grasp.py --model_type diffusion --model_path diffusion.pth --encoder_path checkpoints/object_encoder.pth --sdf_path data/processed/<id>/scene.npz
  ```

Gradient-based optimization (with GQ + nflow prior + FK reg):
- `scripts/optimization/optimize_grasp.py`
  ```bash
  python scripts/optimization/optimize_grasp.py \
    --gq_model_path final_model.pth \
    --nflow_model_path best_nflow_model.pth \
    --data_path data/processed \
    --scene_idx 0 \
    --num_trials 5 --max_iter 100 --lr 1e-2 \
    --fk_regularization_strength 1.0 --log_prob_regularization_strength 0.8
  ```

Optimization using FK loss only:
- `scripts/optimization/optimize_grasp_fk.py`
  ```bash
  python scripts/optimization/optimize_grasp_fk.py --data_path data/processed --scene_idx 0 --num_trials 5 --max_iter 100
  ```

Visualization:
- `scripts/visualizations/vis_grasp.py` (PyBullet GUI)
  ```bash
  python scripts/visualizations/vis_grasp.py data/processed/<id> --filter highest
  # or use your generated outputs: python scripts/visualizations/vis_grasp.py data/output/<id>
  ```
- `scripts/visualizations/preview.py` (level sets + per-slice contours)
  ```bash
  python scripts/visualizations/preview.py -f data/processed/<id>/scene.npz -o data/preview/<id>
  ```
- `scripts/visualizations/vis_sdf_slices.py` (per-slice PNGs with colorbar)
  ```bash
  python scripts/visualizations/vis_sdf_slices.py -f data/processed/<id>/scene.npz -o data/preview_slices/<id>
  ```
- `scripts/visualizations/vis_sdf_target.py` (visualize grasp and near-zero SDF points)
  ```bash
  python scripts/visualizations/vis_sdf_target.py data/processed/<id>/scene.npz --grasp_file_path data/processed/<id>/recording.npz --grasp_index 0
  ```

Utilities:
- `scripts/optimization/find_best_grasp.py` prints the best (lowest) score index in a `recording.npz`.

## Scripts directory structure

Organized by task:
- `scripts/data_processing/`: preprocessing and dataset utilities
- `scripts/training/`: training entrypoints for GQ, AE, NFLOW, Diffusion
- `scripts/testing/`: evaluation and testing utilities
- `scripts/optimization/`: sampling and optimization
- `scripts/visualizations/`: visualization tools

## Troubleshooting

- Always run with `conda activate adlr` to ensure correct Python and deps.
- On macOS, PyBullet GUI needs an interactive session (not over SSH without X).
- If `scikit-image` or `pybullet` not found: your shell is not using the conda env.
- If URDF not found, verify `urdfs/dlr2.urdf` exists and that CWD is repo root.

## Folder structure

- `scripts/`: all CLI utilities (training, preprocessing, visualization, optimization)
- `src/`: models, datasets, kinematics
- `data/`: raw, processed, splits, previews
- `plots/`: generated plots