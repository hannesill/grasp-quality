import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import GQEstimator
from dataset import GraspDataset


class GraspOptimizer:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the grasp optimizer with a pre-trained model.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run optimization on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the model
        self.model = GQEstimator(input_size=48, base_channels=16, fc_dims=[256, 128, 64])
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure gradients are not computed for model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        print("Model loaded successfully")
        
        # Verify gradient setup
        self._verify_gradient_setup()
    
    def _verify_gradient_setup(self):
        """
        Verify that gradients are properly disabled for model parameters.
        """
        model_params_with_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
        total_model_params = sum(1 for p in self.model.parameters())
        
        print(f"Model parameters: {total_model_params} total, {model_params_with_grad} with gradients")
        
        if model_params_with_grad > 0:
            print("WARNING: Some model parameters still have gradients enabled!")
        else:
            print("✓ All model parameters have gradients disabled")
    
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
    
    def predict_grasp_quality(self, sdf, grasp_config):
        """
        Predict grasp quality for given SDF and grasp configuration.
        
        Args:
            sdf: Tensor of shape (48, 48, 48)
            grasp_config: Tensor of shape (19,)
            
        Returns:
            torch.Tensor: Predicted grasp quality score
        """
        # Ensure gradients are not computed for SDF (it's fixed)
        sdf = sdf.detach()
        
        # Encode SDF (gradients disabled for model parameters)
        with torch.no_grad():
            sdf_features = self.model.encode_sdf(sdf)
        
        # Concatenate SDF features with grasp configuration
        # Only grasp_config should have gradients enabled
        combined_features = torch.cat([sdf_features.detach(), grasp_config], dim=0)
        combined_features = combined_features.unsqueeze(0)
        quality = self.model(combined_features)
        
        return quality.squeeze()
    
    def optimize_grasp(self, sdf, initial_grasp=None, learning_rate=0.01, 
                      max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Optimize grasp configuration to maximize predicted grasp quality score.
        
        Args:
            sdf: Object SDF tensor of shape (48, 48, 48)
            initial_grasp: Initial grasp configuration. If None, sample randomly.
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for loss change
            verbose: Whether to print optimization progress
            
        Returns:
            dict: Optimization results containing final grasp, loss history, etc.
        """
        # Initialize grasp configuration
        if initial_grasp is None:
            grasp_config = self.sample_random_grasp()
        else:
            grasp_config = initial_grasp.clone().to(self.device)
        
        # Enable gradient computation for grasp configuration
        grasp_config.requires_grad_(True)
        
        # Setup optimizer for grasp configuration only
        optimizer = torch.optim.Adam([grasp_config], lr=learning_rate)
        
        # Track optimization history
        quality_history = []
        grasp_history = []
        
        if verbose:
            print(f"Starting optimization with {max_iterations} max iterations...")
            initial_quality = self.predict_grasp_quality(sdf, grasp_config).item()
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
        
        prev_quality = float('-inf')  # For maximization, start with negative infinity
        
        # Optimization loop
        for iteration in tqdm(range(max_iterations), disable=not verbose):
            optimizer.zero_grad()
            quality_score = self.predict_grasp_quality(sdf, grasp_config)
            loss = -quality_score  # Minimize negative quality to maximize quality

            loss.backward()
            optimizer.step()
            quality_history.append(quality_score.item())  # Store positive quality, not negative loss
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
        
        if verbose:
            print(f"Final grasp quality: {quality_history[-1]:.6f}")
            print(f"Improvement: {quality_history[-1] - quality_history[0]:.6f}")
        
        return {
            'optimized_grasp': grasp_config.detach().cpu(),
            'initial_grasp': grasp_history[0],
            'quality_history': quality_history,
            'grasp_history': grasp_history,
            'final_quality': quality_history[-1],
            'initial_quality': quality_history[0],
            'improvement': quality_history[-1] - quality_history[0],
            'converged': abs(prev_quality - quality_score.item()) < tolerance,
            'iterations': len(quality_history)
        }
    
    def batch_optimize(self, sdf, num_trials=5, **kwargs):
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
            result = self.optimize_grasp(sdf, **kwargs)
            
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
        plt.title('Grasp Quality Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Grasp Quality Score')
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
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--scene_idx', type=int, default=0, help='Index of scene to optimize for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum optimization iterations')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of random initialization trials')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save visualization plot')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize optimizer
    optimizer = GraspOptimizer(args.model_path, device=args.device)
    
    # Load test data
    data_path = Path(args.data_path)
    dataset = GraspDataset(data_path)
    
    if args.scene_idx >= len(dataset):
        print(f"Scene index {args.scene_idx} out of range. Dataset has {len(dataset)} scenes.")
        return
    
    # Get a test scene
    scene = dataset[args.scene_idx]
    sdf = scene['sdf'].to(optimizer.device)
    
    print(f"Optimizing grasp for scene {args.scene_idx}")
    print(f"SDF shape: {sdf.shape}")
    
    result = optimizer.batch_optimize(
        sdf, 
        num_trials=args.num_trials,
        learning_rate=args.lr,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        verbose=True
    )
    
    print(f"\nOptimization Summary:")
    print(f"Initial quality: {result['initial_quality']:.6f}")
    print(f"Final quality: {result['final_quality']:.6f}")
    print(f"Improvement: {result['improvement']:.6f}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    
    # Visualize results
    optimizer.visualize_optimization(result, save_path=args.save_plot)
    
    print(f"\nOptimized grasp configuration:")
    print(f"Hand pose (pos + quat): {result['optimized_grasp'][:7].tolist()}")
    print(f"Finger joints: {result['optimized_grasp'][7:].tolist()}")


if __name__ == "__main__":
    main() 