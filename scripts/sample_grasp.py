import argparse
import torch
import numpy as np
from pathlib import Path
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_nflow import create_nflow_model
from src.model_diff import DiffusionModel
from src.model import ObjectEncoder


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def sample_grasps_nflow(model, num_samples, device):
    """Samples grasp configurations from the normalizing flow model."""
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples)
    return samples.cpu().numpy()

@torch.no_grad()
def sample_grasps_diffusion(model, object_encoder, sdf, num_samples, timesteps, device):
    """Samples grasp configurations from the diffusion model."""
    model.eval()
    object_encoder.eval()

    sdf_features = object_encoder(sdf.unsqueeze(0)).view(-1)
    sdf_features = sdf_features.repeat(num_samples, 1)

    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    shape = (num_samples, 19) # 19 is the grasp dimension
    grasps = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling grasps', total=timesteps):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        predicted_noise = model(grasps, t, sdf_features)
        
        alpha_t = alphas_cumprod[i]
        alpha_t_prev = alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0, device=device)
        
        noise = torch.randn_like(grasps) if i > 0 else 0

        p1 = 1 / alpha_t.sqrt()
        p2 = (1 - alpha_t) / (1 - alpha_t).sqrt()
        pred_x0 = p1 * (grasps - p2 * predicted_noise)

        # DDIM step
        sigma = 0 # for deterministic sampling
        c1 = (1 - alpha_t_prev - sigma**2).sqrt()
        c2 = (alpha_t_prev - sigma**2).sqrt()

        grasps = c2 * pred_x0 + c1 * predicted_noise + sigma * noise

    return grasps.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Sample grasp configurations from a trained model")
    parser.add_argument('--model_type', type=str, default='nflow', choices=['nflow', 'diffusion'], help='Type of model to use for sampling')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of grasps to sample')
    parser.add_argument('--output_path', type=str, default='data/sampled_grasps', help='Path to save the sampled grasps')
    parser.add_argument('--scene_name', type=str, default='sampled_scene', help='Name for the output scene directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    
    # NFlow specific arguments
    parser.add_argument('--input_dim', type=int, default=19, help='Dimension of the grasp configuration')
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers in the model')
    parser.add_argument('--hidden_features', type=int, default=64, help='Hidden features in the flow transforms')

    # Diffusion specific arguments
    parser.add_argument('--encoder_path', type=str, default="checkpoints/object_encoder.pth", help='Path to pretrained encoder checkpoint')
    parser.add_argument('--sdf_path', type=str, help='Path to the .npz file containing the SDF for diffusion model conditioning')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps for sampling')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for the object encoder')

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.model_type == 'nflow':
        # Load the nflow model
        model = create_nflow_model(
            input_dim=args.input_dim,
            num_layers=args.num_flow_layers,
            hidden_features=args.hidden_features
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("NFlow model loaded successfully")

        # Sample grasps
        print(f"Sampling {args.num_samples} grasps using nflow model...")
        sampled_grasps = sample_grasps_nflow(model, args.num_samples, device)

    elif args.model_type == 'diffusion':
        if not args.sdf_path:
            raise ValueError("An SDF path must be provided for the diffusion model.")

        # Load diffusion model and object encoder
        object_encoder = ObjectEncoder(base_channels=args.base_channels).to(device)
        object_encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        
        model = DiffusionModel().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Diffusion model and object encoder loaded successfully")

        # Get SDF from file
        print(f"Loading SDF from {args.sdf_path}...")
        with np.load(args.sdf_path) as data:
            sdf = data["sdf"]
        sdf = torch.from_numpy(sdf).float().unsqueeze(0).to(device)

        # Sample grasps
        print(f"Sampling {args.num_samples} grasps using diffusion model...")
        sampled_grasps = sample_grasps_diffusion(
            model, object_encoder, sdf, args.num_samples, args.timesteps, device
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
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