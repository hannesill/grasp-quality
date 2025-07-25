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
        self.model = GQEstimator(**model_config)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure gradients are not computed for model parameters
        for param in self.model.parameters():
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

    def verify_grasp_gradients(self, sdf, grasp_config):
        """
        Verify that gradients are being computed correctly for grasp optimization.
        
        Args:
            sdf: Object SDF tensor
            grasp_config: Grasp configuration with requires_grad=True
            
        Returns:
            dict: Gradient verification results
        """
        # Ensure grasp config has gradients enabled
        if not grasp_config.requires_grad:
            raise ValueError("grasp_config must have requires_grad=True for verification")
        
        # Clear any existing gradients
        if grasp_config.grad is not None:
            grasp_config.grad.zero_()
        
        quality = self.predict_grasp_quality(sdf, grasp_config)
        quality.backward()
        
        has_grad = grasp_config.grad is not None
        grad_norm = grasp_config.grad.norm().item() if has_grad else 0.0
        
        # Check model parameters don't have gradients
        model_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        
        results = {
            'grasp_has_gradients': has_grad,
            'grasp_grad_norm': grad_norm,
            'model_params_with_gradients': len(model_grads),
            'gradient_flow_correct': has_grad and len(model_grads) == 0
        }
        
        return results
    
    def sample_random_grasp(self):
        """
        Sample a random grasp configuration.
        For DLR Hand II: 19 dimensions (7 hand pose + 12 finger joints)
        
        Returns:
            torch.Tensor: Random grasp configuration of shape (19,)
        """
        # Hand pose (7 dimensions): position (3) + quaternion (4)
        # Position: sample around object center
        hand_pos = torch.randn(3) * 0.1  # Small random offset from center
        
        # Quaternion: sample random orientation (normalized)
        quaternion = torch.randn(4)
        quaternion = quaternion / torch.norm(quaternion)  # Normalize to unit quaternion
        
        # Finger joints (12 dimensions): sample within reasonable joint limits
        # Assuming joint limits are roughly [0, π] for finger joints
        finger_joints = torch.rand(12) * np.pi
        
        grasp_config = torch.cat([hand_pos, quaternion, finger_joints])
        return grasp_config.to(self.device)

    def sample_grasp_from_nflow(self):
        """
        Sample a grasp configuration from the trained normalizing flow model.
        
        Returns:
            torch.Tensor: Sampled grasp configuration of shape (19,)
        """
        with torch.no_grad():
            grasp_config = self.nflow_model.sample(1).squeeze(0)
        
        return grasp_config.to(self.device)
    
    def predict_grasp_quality(self, sdf_features, grasp_config):
        """
        Predict grasp quality for given SDF and grasp configuration.
        
        Args:
            sdf: Tensor of shape (48, 48, 48)
            grasp_config: Tensor of shape (19,)
            
        Returns:
            torch.Tensor: Predicted grasp quality score
        """
        # Concatenate SDF features with grasp configuration
        # Only grasp_config should have gradients enabled
        combined_features = torch.cat([sdf_features, grasp_config.unsqueeze(0)], dim=1)
        quality = self.model(combined_features)
        
        return quality.squeeze()
    
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

        contact_loss = torch.relu(fingertip_sdf_values).mean() # encourage fingertips to be on the surface (SDF=0)
        collision_loss = torch.relu(-sampled_sdf_values).mean() # penalize all points for being inside the object (SDF<0)

        return collision_loss + contact_loss

    def optimize_grasp(self, sdf, scale, translation, initial_grasp=None, learning_rate=0.01, 
                      max_iterations=100, tolerance=1e-6, verbose=True,
                      log_prob_regularization_strength=1e-4, fk_regularization_strength=1.0):
        """
        Optimize grasp configuration to maximize predicted grasp quality score.
        
        Args:
            sdf: Object SDF tensor of shape (48, 48, 48)
            initial_grasp: Initial grasp configuration. If None, sample randomly.
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for loss change
            verbose: Whether to print optimization progress
            log_prob_regularization_strength: Strength of distribution-based regularization
            fk_regularization_strength: Strength of hand-object interaction regularization
            
        Returns:
            dict: Optimization results containing final grasp, loss history, etc.
        """
        # Initialize grasp configuration
        if initial_grasp is None:
            grasp_config = self.sample_grasp_from_nflow()
        else:
            grasp_config = initial_grasp.clone().to(self.device)
        
        grasp_config.requires_grad_(True)

        sdf = sdf.detach()
        with torch.no_grad():
            sdf_features = self.model.encode_sdf(sdf.unsqueeze(0))
            sdf_features = sdf_features.detach()
        
        # Setup optimizer for grasp configuration only
        optimizer = torch.optim.Adam([grasp_config], lr=learning_rate)
        
        # Track optimization history
        quality_history = []
        log_prob_history = []
        fk_loss_history = []
        grasp_history = []
        
        if verbose:
            print(f"Starting optimization with {max_iterations} max iterations...")
            initial_quality = self.predict_grasp_quality(sdf_features, grasp_config).item()
            print(f"Initial grasp quality: {initial_quality:.6f}")
            
            # Verify gradient computation on first iteration
            print("Verifying gradient computation...")
            grad_check = self.verify_grasp_gradients(sdf, grasp_config.clone().detach().requires_grad_(True))
            print(f"✓ Grasp gradients: {grad_check['grasp_has_gradients']}")
            print(f"✓ Gradient norm: {grad_check['grasp_grad_norm']:.6f}")
            print(f"✓ Model params with gradients: {grad_check['model_params_with_gradients']}")
            print(f"✓ Gradient flow correct: {grad_check['gradient_flow_correct']}")
            if not grad_check['gradient_flow_correct']:
                raise RuntimeError("Gradient computation is not set up correctly!")
        
        prev_quality = float('-inf')
        
        # Optimization loop
        for iteration in tqdm(range(max_iterations)):
            optimizer.zero_grad()
            quality_score = self.predict_grasp_quality(sdf_features, grasp_config).squeeze()
            
            log_prob = self.nflow_model.log_prob(grasp_config.unsqueeze(0)).squeeze()
            fk_loss = self.compute_fk_loss(grasp_config, sdf, scale, translation)
            
            loss = (
                - quality_score 
                - log_prob_regularization_strength * log_prob 
                + fk_regularization_strength * fk_loss
            )

            loss.backward()
            optimizer.step()
            quality_history.append(quality_score.item())
            log_prob_history.append(log_prob_regularization_strength * log_prob.item())
            fk_loss_history.append(fk_regularization_strength * fk_loss.item())
            grasp_history.append(grasp_config.detach().clone().cpu())

            # Check for convergence
            current_quality = quality_score.item()
            if abs(prev_quality - current_quality) < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration+1}")
                break
            
            prev_quality = current_quality
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: Quality score = {current_quality:.6f}")
        
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
            'converged': abs(prev_quality - quality_score.item()) < tolerance,
            'iterations': len(quality_history)
        }
    
    def batch_optimize(self, sdf, scale, translation, num_trials=5, **kwargs):
        """
        Run multiple optimization trials and return the best result.
        
        Args:
            sdf: Object SDF tensor
            num_trials: Number of random initialization trials
            **kwargs: Additional arguments passed to optimize_grasp
            
        Returns:
            dict: Best optimization result among all trials
        """
        best_result = None
        best_quality = float('-inf')
        
        print(f"Running {num_trials} optimization trials...")
        
        for trial in range(num_trials):
            print(f"\n--- Trial {trial+1}/{num_trials} ---")
            result = self.optimize_grasp(sdf, scale, translation, **kwargs)
            
            if result['final_quality'] > best_quality:
                best_quality = result['final_quality']
                best_result = result
                best_result['trial'] = trial + 1
        
        print(f"\nBest result from trial {best_result['trial']} with quality: {best_quality:.6f}")
        return best_result
    
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
        plt.plot(result['quality_history'])
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
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained GQ model weights')
    parser.add_argument('--model_config', type=json.loads, default={}, help='Config for GQ model')
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
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save visualization plot')
    parser.add_argument('--save_path', type=str, default='data/output', help='Path to save optimized grasps')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose output')
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = {'input_size': 48, 'base_channels': 8, 'fc_dims': [512, 128, 64]}
    model_config.update(args.model_config)

    nflow_model_config = {'input_dim': 19}
    nflow_model_config.update(args.nflow_model_config)
    
    # Initialize optimizer
    optimizer = GraspOptimizer(
        args.model_path, 
        model_config, 
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
    
    result = optimizer.batch_optimize(
        sdf,
        scale,
        translation,
        num_trials=args.num_trials,
        learning_rate=args.lr,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        verbose=args.verbose,
        log_prob_regularization_strength=args.log_prob_regularization_strength,
        fk_regularization_strength=args.fk_regularization_strength,
    )

    result['optimized_grasp'][:3] = result['optimized_grasp'][:3] / scale + translation
    
    print(f"\nOptimization Summary:")
    print(f"Initial quality: {result['initial_quality']:.6f}")
    print(f"Final quality: {result['final_quality']:.6f}")
    print(f"Improvement: {result['improvement']:.6f}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    
    # Visualize results
    optimizer.visualize_optimization(result, save_path=args.save_plot)

    # Save the optimized grasp configuration
    scene_name = Path(dataset.data_files[args.scene_idx]).parent.name
    raw_dir = Path('data/raw')
    raw_path = raw_dir / scene_name
    output_dir = Path(args.save_path)
    output_path = output_dir / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving optimized grasp to {output_path}")

    # Copy mesh.obj
    source_mesh_path = raw_path / 'mesh.obj'
    if source_mesh_path.exists():
        shutil.copy(source_mesh_path, output_path)
    else:
        print(f"Warning: mesh.obj not found at {source_mesh_path}")

    # Save grasp and score
    optimized_grasp = result['optimized_grasp'].numpy()
    final_quality = result['final_quality']
    
    # Save in the format expected by vis_grasp.py (as arrays)
    grasps_to_save = np.array([optimized_grasp])
    scores_to_save = np.array([final_quality])
    
    npz_path = output_path / 'recording.npz'
    np.savez(npz_path, grasps=grasps_to_save, scores=scores_to_save)
    
    print(f"Saved recording.npz with 1 grasp.")
    print(f"To visualize, run: python3 vis_grasp.py {output_path}")


if __name__ == "__main__":
    main() 