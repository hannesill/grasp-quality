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
from src.fk import DLRHandFK

class GraspOptimizer:
    def __init__(self, urdf_path, device='cuda'):
        """
        Initialize the grasp optimizer.
        
        Args:
            urdf_path: Path to the URDF file for the hand
            device: Device to run optimization on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize forward kinematics
        self.fk = DLRHandFK(urdf_path, device=self.device)

        print("Forward kinematics initialized successfully")
    
    def sample_grasp(self):
        """
        Sample a random grasp configuration.
        
        Returns:
            torch.Tensor: Sampled grasp configuration of shape (19,)
        """
        hand_pos = torch.randn(3) * 0.1 # hand position (x, y, z)
        quaternion = torch.randn(4) # hand orientation (qx, qy, qz, qw)
        quaternion = quaternion / torch.norm(quaternion)
        finger_joints = torch.rand(12) * np.pi # 4 x 3 finger joint angles (radians)
        
        grasp_config = torch.cat([hand_pos, quaternion, finger_joints])
            
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
                      max_iterations=100, tolerance=1e-6, fk_regularization_strength=1.0,
                      joint_regularization_strength=0.0):
        """
        Optimize grasp configuration to minimize forward kinematics loss.
        
        Args:
            sdf: Object SDF tensor of shape (48, 48, 48)
            initial_grasp: Initial grasp configuration. If None, sample randomly.
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for loss change
            fk_regularization_strength: Strength of hand-object interaction regularization
            joint_regularization_strength: Strength of joint angle penalty
            
        Returns:
            dict: Optimization results containing final grasp, loss history, etc.
        """
        # Sample initial grasp configuration
        if initial_grasp is None:
            grasp_config = self.sample_grasp()
        else:
            grasp_config = initial_grasp.clone().to(self.device)
        
        grasp_config.requires_grad_(True)
        optimizer = torch.optim.Adam([grasp_config], lr=learning_rate)

        sdf = sdf.detach()
        
        fk_loss_history = []
        grasp_history = []
        
        prev_loss = float('inf')
        
        # Optimization loop
        pbar = tqdm(range(max_iterations), desc="Optimizing grasp")
        for step in pbar:
            optimizer.zero_grad()

            fk_loss = self.compute_fk_loss(grasp_config, sdf, scale, translation)
            
            # Penalize large joint angles for the 3 main fingers (9 joints)
            # The thumb and palm base joints are not penalized.
            # The last 12 parameters of grasp_config are the joint angles.
            joint_angles = grasp_config[7:]
            
            # As per fk.py, the first 9 joint angles correspond to the ring, middle, 
            # and fore fingers. The last 3 are for the thumb.
            main_finger_joints = joint_angles[:9]
            
            joint_loss = torch.sum(main_finger_joints**2)
            
            loss = (fk_regularization_strength * fk_loss + 
                    joint_regularization_strength * joint_loss)

            loss.backward()
            optimizer.step()

            fk_loss_history.append(fk_loss.clone().detach().cpu().item())
            grasp_history.append(grasp_config.clone().detach().cpu())

            # Check for convergence
            current_loss = loss.item()
            if abs(prev_loss - current_loss) < tolerance:
                break
            prev_loss = current_loss

            pbar.set_postfix(loss=f'{current_loss:.4f}', fk_loss=f'{fk_loss.item():.4f}')
        
        print(f"Final FK loss: {fk_loss_history[-1]:.6f}")
        print(f"Improvement: {fk_loss_history[0] - fk_loss_history[-1]:.6f}")
        
        return {
            'optimized_grasp': grasp_config.detach().cpu(),
            'initial_grasp': grasp_history[0],
            'fk_loss_history': fk_loss_history,
            'grasp_history': grasp_history,
            'final_fk_loss': fk_loss_history[-1],
            'initial_fk_loss': fk_loss_history[0],
            'improvement': fk_loss_history[0] - fk_loss_history[-1],
            'converged': abs(prev_loss - current_loss) < tolerance,
            'iterations': len(fk_loss_history)
        }
    
    def batch_optimize(self, sdf, scale, translation, num_trials=5, **kwargs):
        """
        Run multiple optimization trials and return all results, sorted by final FK loss.
        
        Args:
            sdf: Object SDF tensor
            num_trials: Number of random initialization trials
            **kwargs: Additional arguments passed to optimize_grasp
            
        Returns:
            list: A list of optimization result dictionaries, sorted by final_fk_loss.
        """
        all_results = []
        
        print(f"Running {num_trials} optimization trials...")
        
        for trial in range(num_trials):
            print(f"\n--- Trial {trial+1}/{num_trials} ---")
            result = self.optimize_grasp(sdf, scale, translation, **kwargs)
            result['trial'] = trial + 1
            all_results.append(result)

        # Sort results by final fk_loss in ascending order
        all_results.sort(key=lambda r: r['final_fk_loss'])

        if all_results:
            print(f"\nBest result from trial {all_results[0]['trial']} with FK loss: {all_results[0]['final_fk_loss']:.6f}")
        
        return all_results
    
    def visualize_optimization(self, result, save_path=None):
        """
        Visualize the optimization progress with detailed plots for loss,
        pose, and joint angles.
        
        Args:
            result: Result dictionary from optimize_grasp
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(20, 5))
        
        # 1. Plot loss history
        plt.subplot(1, 3, 1)
        plt.plot(result['fk_loss_history'], label='FK Loss')
        plt.title('Grasp FK Loss Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('FK Loss')
        plt.legend()
        plt.grid(True)
        
        grasp_array = torch.stack(result['grasp_history']).numpy()
        pose_labels = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
        joint_labels = [
            'ring_prox', 'ring_knuck', 'ring_mid', 
            'mid_prox', 'mid_knuck', 'mid_mid', 
            'fore_prox', 'fore_knuck', 'fore_mid', 
            'thumb_prox', 'thumb_knuck', 'thumb_mid'
        ]

        # 2. Plot pose configuration evolution
        plt.subplot(1, 3, 2)
        for i in range(7):
            plt.plot(grasp_array[:, i], label=pose_labels[i])
        plt.title('Pose Evolution (Position & Rotation)')
        plt.xlabel('Iteration')
        plt.ylabel('Values')
        plt.legend(fontsize='small')
        plt.grid(True)

        # 3. Plot joint angle evolution
        plt.subplot(1, 3, 3)
        for i in range(12):
            plt.plot(grasp_array[:, 7 + i], label=joint_labels[i])
        plt.title('Joint Angle Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Radians')
        plt.legend(fontsize='small')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize grasp configurations using gradient descent based on FK loss")
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--scene_idx', type=int, default=0, help='Index of scene to optimize for')
    parser.add_argument('--sdf_file', type=str, default=None, help='Path to a specific SDF file to use instead of the dataset')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum optimization iterations')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of random initialization trials')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--fk_regularization_strength', type=float, default=1.0, help='Strength of hand-object interaction regularization')
    parser.add_argument('--joint_regularization_strength', type=float, default=0.0, help='Strength of joint angle penalty')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save visualization plot')
    parser.add_argument('--save_path', type=str, default='data/output', help='Path to save optimized grasps')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize optimizer
    optimizer = GraspOptimizer(
        'urdfs/dlr2.urdf', 
        device=args.device
    )
    
    # Load test data
    if args.sdf_file:
        data = np.load(args.sdf_file)
        sdf = torch.from_numpy(data['sdf']).float().to(optimizer.device)
        scale = 1.0  # Default scale
        translation = torch.tensor([0.0, 0.0, 0.0])  # Default translation
        scene_name = Path(args.sdf_file).stem
        print(f"Loaded SDF from {args.sdf_file}")
    else:
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
        scene_name = Path(dataset.data_files[args.scene_idx]).parent.name
        print(f"Optimizing grasp for scene {args.scene_idx} from dataset")
    print(f"SDF shape: {sdf.shape}")
    
    results = optimizer.batch_optimize(
        sdf,
        scale,
        translation,
        num_trials=args.num_trials,
        learning_rate=args.lr,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        fk_regularization_strength=args.fk_regularization_strength,
        joint_regularization_strength=args.joint_regularization_strength,
    )

    if not results:
        print("Optimization did not produce any results.")
        return

    best_result = results[0]
    
    print(f"\nOptimization Summary (Best Trial):")
    print(f"Initial FK loss: {best_result['initial_fk_loss']:.6f}")
    print(f"Final FK loss: {best_result['final_fk_loss']:.6f}")
    print(f"Improvement: {best_result['improvement']:.6f}")
    print(f"Converged: {best_result['converged']}")
    print(f"Iterations: {best_result['iterations']}")
    
    # Visualize results
    optimizer.visualize_optimization(best_result, save_path=args.save_plot)

    # Save the optimized grasp configuration
    if not args.sdf_file:
        raw_dir = Path('data/raw')
        raw_path = raw_dir / scene_name
        output_dir = Path(args.save_path)
        output_path = output_dir / scene_name
        output_path.mkdir(parents=True, exist_ok=True)
        # Copy mesh.obj
        source_mesh_path = raw_path / 'mesh.obj'
        if source_mesh_path.exists():
            shutil.copy(source_mesh_path, output_path)
        else:
            print(f"Warning: mesh.obj not found at {source_mesh_path}")
    else:
        output_dir = Path(args.save_path)
        output_path = output_dir / scene_name
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving optimized grasps to {output_path}")

    # Save grasp and score
    all_grasps_to_save = []
    all_scores_to_save = []
    
    num_steps_to_save = 10
    
    for result in results:
        grasp_history = result['grasp_history']
        fk_loss_history = result['fk_loss_history']
    
        num_iterations = len(grasp_history)
    
        if num_iterations > 1:
            indices = np.linspace(0, num_iterations - 1, num_steps_to_save, dtype=int)
        else:
            indices = np.array([0]) if num_iterations > 0 else np.array([])

        for i in indices:
            grasp = grasp_history[i].clone()
            grasp[:3] = grasp[:3] / scale + translation
            all_grasps_to_save.append(grasp.numpy())
            all_scores_to_save.append(fk_loss_history[i])
    
    # Save in the format expected by vis_grasp.py (as arrays)
    grasps_to_save = np.array(all_grasps_to_save)
    scores_to_save = np.array(all_scores_to_save)
    
    npz_path = output_path / 'recording.npz'
    np.savez(npz_path, grasps=grasps_to_save, scores=scores_to_save)
    
    print(f"Saved recording.npz with {len(grasps_to_save)} grasp(s) from {len(results)} trials.")
    print(f"To visualize, run: python3 scripts/vis_grasp.py {output_path}")


if __name__ == "__main__":
    main()

