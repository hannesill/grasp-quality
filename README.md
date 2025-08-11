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

## Dataset Setup

The dataset is private. You need to get access to the dataset and copy it under `data/raw`. Also, the URDF of the robot hand in the dataset is private. You need to copy it under `urdfs/`.

Expected directories:
- `data/raw/<scene_id>/{mesh.obj, recording.npz}`: raw mesh and original grasps+scores
- `urdfs/dlr2.urdf`

## Optional quick sanity check

Create a toy SDF and render slices:
```bash
conda activate adlr
python scripts/data_processing/create_sphere_sdf.py
python scripts/visualizations/vis_sdf_slices.py -f data/sphere.npz
```

You should see the 48 SDF slices in `data/preview_slices`. This toy SDF can be a useful resource for sanity checks later on.

## Scripts overview and usage

### /data_processing/

Preprocessing:
- `scripts/data_processing/preprocess.py`
  ```bash
  python scripts/data_processing/preprocess.py \
    -r data/raw \
    -o data/processed \
    -g 48 \
    --mesh_max_extent 0   # optional; auto-computed if omitted # TODO Lucas: 0 correct?
  ```
  Produces `scene.npz` files and writes `mesh_max_extent.txt` in output. There is a `--fix_mesh` flag that signals the preprocesser to try to fix non-watertight meshes. However, this led to unwanted artifacts in the SDFs for us, which is why we leave it out. The downside of this is, that there are no negative SDF values, which prevents us from using collision detection. This is a bug in the mesh2sdf library.

Splits:
- `scripts/data_processing/create_splits.py`
  ```bash
  python scripts/data_processing/create_splits.py --data_path data/processed --train_ratio 0.8 --val_ratio 0.1 --output_path data/splits.json
  ```

We used a train/val/test split of 80/10/10.

Expected directories after data processing
- `data/processed/<scene_id>/scene.npz`: produced by preprocessing; contains `sdf, grasps, scores, translation, global_scale`
- `data/splits.json`: train/val/test split ids

### /training/

Training GQ model:
- `scripts/training/train_gq.py`
  ```bash
  python scripts/training/train_gq.py \
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

  We tried training an autoencoder to use its encoder in the GQ model. However, this resulted in a noticably worse validation loss compared to end-to-end training. Because we have over 7M samples, end-to-end training makes most sense.

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

### /testing/ 

Evaluation and plots:
- `scripts/testing/eval_gq_model_plot.py` (targets-vs-preds plot)
  ```bash
  python scripts/testing/eval_gq_model_plot.py --model_path final_model.pth --split test --output plots/gq_ordering_final.png
  ```

### /optimization/

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

Find best grasp from optimizer
- `scripts/optimization/find_best_grasp.py` prints the best (lowest) score index in a `recording.npz`.

### /visualization/
- `scripts/visualizations/vis_grasp.py` (PyBullet GUI)
  ```bash
  python scripts/visualizations/vis_grasp.py data/processed/<id> --filter highest
  # or use your generated outputs: python scripts/visualizations/vis_grasp.py data/output/<id>
  ```
- `scripts/visualizations/vis_sdf_slices.py` (per-slice PNGs with colorbar)
  ```bash
  python scripts/visualizations/vis_sdf_slices.py -f data/processed/<id>/scene.npz -o data/preview_slices/<id>
  ```

## Scripts directory structure

Organized by task:
- `scripts/data_processing/`: preprocessing and dataset utilities
- `scripts/training/`: training entrypoints for GQ, AE, NFLOW, Diffusion
- `scripts/testing/`: evaluation and testing utilities
- `scripts/optimization/`: sampling and optimization
- `scripts/visualizations/`: visualization tools

## Troubleshooting

- Always run with `conda activate adlr` to ensure correct Python and deps.
- On macOS, PyBullet GUI needs an interactive session

## Folder structure

- `scripts/`: all CLI utilities (training, preprocessing, visualization, optimization)
- `src/`: models, datasets, kinematics
- `data/`: raw, processed, splits, previews
- `plots/`: generated plots
- `docs/`: reports, poster etc.