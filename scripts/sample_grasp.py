import argparse
import torch
import numpy as np
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_nflow import create_nflow_model

def sample_grasps(model, num_samples, device):
    """Samples grasp configurations from the normalizing flow model."""
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples)
    return samples.cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Sample grasp configurations from a trained normalizing flow model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained nflow model weights')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of grasps to sample')
    parser.add_argument('--output_path', type=str, default='data/sampled_grasps', help='Path to save the sampled grasps')
    parser.add_argument('--scene_name', type=str, default='sampled_scene', help='Name for the output scene directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--input_dim', type=int, default=19, help='Dimension of the grasp configuration')
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers in the model')
    parser.add_argument('--hidden_features', type=int, default=64, help='Hidden features in the flow transforms')
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = create_nflow_model(
        input_dim=args.input_dim,
        num_layers=args.num_flow_layers,
        hidden_features=args.hidden_features
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully")

    # Sample grasps
    print(f"Sampling {args.num_samples} grasps...")
    sampled_grasps = sample_grasps(model, args.num_samples, device)
    
    # Save the sampled grasps
    output_dir = Path(args.output_path) / args.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = output_dir / 'recording.npz'
    # The visualization script expects scores, so we'll add dummy scores.
    dummy_scores = np.ones(args.num_samples) 
    
    np.savez(npz_path, grasps=sampled_grasps, scores=dummy_scores)

    print(f"Saved {args.num_samples} sampled grasps to {npz_path}")
    print(f"To visualize, you might need a mesh file in the same directory.")
    print(f"Example visualization command: python3 vis_grasp.py {output_dir}")

if __name__ == "__main__":
    main()