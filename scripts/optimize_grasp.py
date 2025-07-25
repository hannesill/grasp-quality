import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import GraspDataset
from src.model import GQEstimator
from src.model_nflow import create_nflow_model
from src.fk import DLRHandFK

class GraspOptimizer:
    def __init__(self, model_path, model_config, nflow_model_path, nflow_model_config, urdf_path, device='cuda'):
        """
        Initialize the grasp optimizer with a pre-trained model.
        
        Args:
            model_path: Path to the trained model weights
            nflow_model_path: Path to the trained nflow model weights
            urdf_path: Path to the URDF file for the hand
            device: Device to run optimization on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the GQ model
        weights = torch.load(model_path, map_location=self.device)
        # weights = {k.replace('_orig_mod.', ''): v for k, v in weights.items()}
        # torch.save(weights, model_path)
        self.gq_model = GQEstimator(**model_config)
        self.gq_model.load_state_dict(weights)
        self.gq_model.to(self.device)
        self.gq_model.eval()
        
        # Ensure gradients are not computed for model parameters
        for param in self.gq_model.parameters():
            param.requires_grad = False
        
        print("GQ Model loaded successfully")
        
        # Load the nflow model
        self.nflow_model = create_nflow_model(**nflow_model_config)
        self.nflow_model.load_state_dict(torch.load(nflow_model_path, map_location=self.device))
        self.nflow_model.to(self.device)
        self.nflow_model.eval()

        # Ensure gradients are not computed for nflow model parameters
        for param in self.nflow_model.parameters():
            param.requires_grad = False
        
        print("NFlow Model loaded successfully")
        
        # Initialize forward kinematics
        self.fk = DLRHandFK(urdf_path, device=self.device)

        print("Forward kinematics initialized successfully")
    
    def sample_grasp(self, method='nflow'):
        """
        Sample a grasp configuration.
        
        Args:
            method (str): 'nflow' to sample from the normalizing flow model,
                          'random' to sample a random grasp configuration.
        
        Returns:
            torch.Tensor: Sampled grasp configuration of shape (19,)
        """
        if method == 'nflow':
            with torch.no_grad():
                grasp_config = self.nflow_model.sample(1).squeeze(0)
        elif method == 'random':
            hand_pos = torch.randn(3) * 0.1 # hand position (x, y, z)
            quaternion = torch.randn(4) # hand orientation (qx, qy, qz, qw)
            quaternion = quaternion / torch.norm(quaternion)
            finger_joints = torch.rand(12) * np.pi # 4 x 3 finger joint angles (radians)
            
            grasp_config = torch.cat([hand_pos, quaternion, finger_joints])
        else:
            raise ValueError(f"Unknown grasp sampling method: {method}")
            
        return grasp_config.to(self.device)
    
    def compute_fk_loss(self, grasp_config, sdf, scale, translation):
        """
        Computes a regularization loss based on hand-object interaction.
        - Penalizes collisions (control points inside the object).
        - Encourages fingertips to be close to the object surface.
        """
        # Create a new tensor for FK with the grasp in the world frame
        # to avoid in-place modification of the input tensor.
        grasp_config_world = grasp_config.clone()
        grasp_config_world[:3] = grasp_config[:3] / scale + translation
        control_points = self.fk.forward(grasp_config_world) # (N, 3) tensor in the world frame
        points_normalized = (control_points - translation) / scale # normalize to [-1, 1]
        points_for_sampling = points_normalized.view(1, -1, 1, 1, 3)

        sampled_sdf_values = F.grid_sample(
            sdf.unsqueeze(0).unsqueeze(0), 
            points_for_sampling,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        sampled_sdf_values = sampled_sdf_values.squeeze()
        fingertip_sdf_values = sampled_sdf_values[:4]

        contact_loss = torch.square(torch.relu(fingertip_sdf_values)).mean() # encourage fingertips to be on the surface (SDF=0)
        collision_loss = torch.square(torch.relu(-sampled_sdf_values)).mean() # penalize all points for being inside the object (SDF<0)

        return collision_loss + contact_loss

    def optimize_grasp(self, sdf, scale, translation, initial_grasp=None, learning_rate=0.01, 
                      max_iterations=100, tolerance=1e-6,
                      log_prob_regularization_strength=1e-4, fk_regularization_strength=1.0,
                      sampling_method='nflow'):
        """
        Optimize grasp configuration to maximize predicted grasp quality score.
        
        Args:
            sdf: Object SDF tensor of shape (48, 48, 48)
            initial_grasp: Initial grasp configuration. If None, sample randomly.
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for loss change
            log_prob_regularization_strength: Strength of distribution-based regularization
            fk_regularization_strength: Strength of hand-object interaction regularization
            sampling_method: Method for sampling initial grasp ('nflow' or 'random')
            
        Returns:
            dict: Optimization results containing final grasp, loss history, etc.
        """
        # Sample initial grasp configuration
        if initial_grasp is None:
            grasp_config = self.sample_grasp(method=sampling_method)
        else:
            grasp_config = initial_grasp.clone().to(self.device)
        
        grasp_config.requires_grad_(True)
        optimizer = torch.optim.Adam([grasp_config], lr=learning_rate)

        # Precompute SDF features
        sdf = sdf.detach()
        with torch.no_grad():
            sdf_features = self.gq_model.encode_sdf(sdf.unsqueeze(0))
            sdf_features = sdf_features.detach()
        
        quality_history = []
        log_prob_history = []
        fk_loss_history = []
        grasp_history = []
        
        prev_loss = float('-inf')
        
        # Optimization loop
        pbar = tqdm(range(max_iterations), desc="Optimizing grasp")
        for step in pbar:
            optimizer.zero_grad()

            combined_features = torch.cat([sdf_features, grasp_config.unsqueeze(0)], dim=1)
            quality_score = self.gq_model(combined_features).squeeze()
            log_prob = self.nflow_model.log_prob(grasp_config.unsqueeze(0)).squeeze()
            fk_loss = self.compute_fk_loss(grasp_config, sdf, scale, translation)
            
            loss = (
                - quality_score 
                - log_prob_regularization_strength * log_prob 
                + fk_regularization_strength * fk_loss
            )

            loss.backward()
            optimizer.step()

            quality_history.append(quality_score.clone().detach().cpu().item())
            log_prob_history.append(log_prob_regularization_strength * log_prob.clone().detach().cpu().item())
            fk_loss_history.append(fk_regularization_strength * fk_loss.clone().detach().cpu().item())
            grasp_history.append(grasp_config.clone().detach().cpu())

            # Check for convergence
            current_loss = loss.item()
            if abs(prev_loss - current_loss) < tolerance:
                break
            prev_loss = current_loss

            pbar.set_postfix(loss=f'{current_loss:.4f}', quality=f'{quality_score.item():.4f}')
        
        print(f"Final grasp quality: {quality_history[-1]:.6f}")
        print(f"Improvement: {quality_history[-1] - quality_history[0]:.6f}")
        
        return {
            'optimized_grasp': grasp_config.detach().cpu(),
            'initial_grasp': grasp_history[0],
            'quality_history': quality_history,
            'log_prob_history': log_prob_history,
            'fk_loss_history': fk_loss_history,
            'grasp_history': grasp_history,
            'final_quality': quality_history[-1],
            'initial_quality': quality_history[0],
            'improvement': quality_history[-1] - quality_history[0],
            'converged': abs(prev_loss - current_loss) < tolerance,
            'iterations': len(quality_history)
        }
    
    def batch_optimize(self, sdf, scale, translation, num_trials=5, **kwargs):
        """
        Run multiple optimization trials and return all results, sorted by final quality.
        
        Args:
            sdf: Object SDF tensor
            num_trials: Number of random initialization trials
            **kwargs: Additional arguments passed to optimize_grasp
            
        Returns:
            list: A list of optimization result dictionaries, sorted by final_quality.
        """
        all_results = []
        
        print(f"Running {num_trials} optimization trials...")
        
        for trial in range(num_trials):
            print(f"\n--- Trial {trial+1}/{num_trials} ---")
            result = self.optimize_grasp(sdf, scale, translation, **kwargs)
            result['trial'] = trial + 1
            all_results.append(result)

        # Sort results by final quality in descending order
        all_results.sort(key=lambda r: r['final_quality'], reverse=True)

        if all_results:
            print(f"\nBest result from trial {all_results[0]['trial']} with quality: {all_results[0]['final_quality']:.6f}")
        
        return all_results
    
    def visualize_optimization(self, result, save_path=None):
        """
        Visualize the optimization progress.
        
        Args:
            result: Result dictionary from optimize_grasp
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss history
        plt.subplot(1, 2, 1)
        plt.plot(result['quality_history'], label='Quality Score')
        plt.plot(result['log_prob_history'], label='Log Probability')
        plt.plot(result['fk_loss_history'], label='FK Loss')
        plt.title('Grasp Quality Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Grasp Quality Score')
        plt.legend()
        plt.grid(True)
        
        # Plot grasp configuration evolution (first few dimensions)
        plt.subplot(1, 2, 2)
        grasp_array = torch.stack(result['grasp_history']).numpy()
        for i in range(min(6, grasp_array.shape[1])):  # Plot first 6 dimensions
            plt.plot(grasp_array[:, i], label=f'Dim {i}', alpha=0.7)
        plt.title('Grasp Configuration Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Configuration Values')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize grasp configurations using gradient descent")
    parser.add_argument('--gq_model_path', type=str, required=True, help='Path to trained GQ model weights')
    parser.add_argument('--gq_model_config', type=json.loads, default={}, help='Config for GQ model')
    parser.add_argument('--nflow_model_path', type=str, required=True, help='Path to trained nflow model weights')
    parser.add_argument('--nflow_model_config', type=json.loads, default={}, help='Config for nflow model. Partial configs will be merged with defaults.')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--scene_idx', type=int, default=0, help='Index of scene to optimize for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum optimization iterations')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of random initialization trials')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--log_prob_regularization_strength', type=float, default=0.8, help='Strength of distribution-based regularization')
    parser.add_argument('--fk_regularization_strength', type=float, default=1.0, help='Strength of hand-object interaction regularization')
    parser.add_argument('--sampling_method', type=str, default='nflow', choices=['nflow', 'random'], help='Method for sampling initial grasps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save visualization plot')
    parser.add_argument('--save_path', type=str, default='data/output', help='Path to save optimized grasps')
    return parser.parse_args()


def main():
    args = parse_args()

    gq_model_config = {'input_size': 48, 'base_channels': 8, 'fc_dims': [512, 128, 64]}
    gq_model_config.update(args.gq_model_config)

    nflow_model_config = {'input_dim': 19}
    nflow_model_config.update(args.nflow_model_config)
    
    # Initialize optimizer
    optimizer = GraspOptimizer(
        args.gq_model_path, 
        gq_model_config, 
        args.nflow_model_path, 
        nflow_model_config, 
        'urdfs/dlr2.urdf', 
        device=args.device
    )
    
    # Load test data
    data_path = Path(args.data_path)
    dataset = GraspDataset(data_path, preload=False)
    
    if args.scene_idx >= len(dataset):
        print(f"Scene index {args.scene_idx} out of range. Dataset has {len(dataset)} scenes.")
        return
    
    # Get a test scene
    scene = dataset[args.scene_idx]
    sdf = scene['sdf'].to(optimizer.device)
    scale = scene['scale']
    translation = scene['translation']
    
    print(f"Optimizing grasp for scene {args.scene_idx}")
    print(f"SDF shape: {sdf.shape}")
    
    results = optimizer.batch_optimize(
        sdf,
        scale,
        translation,
        num_trials=args.num_trials,
        learning_rate=args.lr,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        log_prob_regularization_strength=args.log_prob_regularization_strength,
        fk_regularization_strength=args.fk_regularization_strength,
        sampling_method=args.sampling_method
    )

    if not results:
        print("Optimization did not produce any results.")
        return

    best_result = results[0]
    
    print(f"\nOptimization Summary (Best Trial):")
    print(f"Initial quality: {best_result['initial_quality']:.6f}")
    print(f"Final quality: {best_result['final_quality']:.6f}")
    print(f"Improvement: {best_result['improvement']:.6f}")
    print(f"Converged: {best_result['converged']}")
    print(f"Iterations: {best_result['iterations']}")
    
    # Visualize results
    optimizer.visualize_optimization(best_result, save_path=args.save_plot)

    # Save the optimized grasp configuration
    scene_name = Path(dataset.data_files[args.scene_idx]).parent.name
    raw_dir = Path('data/raw')
    raw_path = raw_dir / scene_name
    output_dir = Path(args.save_path)
    output_path = output_dir / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving optimized grasps to {output_path}")

    # Copy mesh.obj
    source_mesh_path = raw_path / 'mesh.obj'
    if source_mesh_path.exists():
        shutil.copy(source_mesh_path, output_path)
    else:
        print(f"Warning: mesh.obj not found at {source_mesh_path}")

    # Save grasp and score
    all_grasps_to_save = []
    all_scores_to_save = []
    
    num_steps_to_save = 10
    
    for result in results:
        grasp_history = result['grasp_history']
        quality_history = result['quality_history']
    
        num_iterations = len(grasp_history)
    
        if num_iterations > 1:
            indices = np.linspace(0, num_iterations - 1, num_steps_to_save, dtype=int)
        else:
            indices = np.array([0]) if num_iterations > 0 else np.array([])

        for i in indices:
            grasp = grasp_history[i].clone()
            grasp[:3] = grasp[:3] / scale + translation
            all_grasps_to_save.append(grasp.numpy())
            all_scores_to_save.append(quality_history[i])
    
    # Save in the format expected by vis_grasp.py (as arrays)
    grasps_to_save = np.array(all_grasps_to_save)
    scores_to_save = np.array(all_scores_to_save)
    
    npz_path = output_path / 'recording.npz'
    np.savez(npz_path, grasps=grasps_to_save, scores=scores_to_save)
    
    print(f"Saved recording.npz with {len(grasps_to_save)} grasp(s) from {len(results)} trials.")
    print(f"To visualize, run: python3 scripts/vis_grasp.py {output_path}")


if __name__ == "__main__":
    main() 