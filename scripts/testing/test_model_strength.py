#!/usr/bin/env python3
"""
Comprehensive test file for model predictions and grasp visualization.

This script evaluates model prediction strength by calculating:
- Top-K accuracy rates (Top1, Top5, Top10, Top50) 
- Score ordering correlation (Spearman, Kendall)
- Visualizes top 10 grasps with highest scores

Usage:
    python test_model_strength.py --model_path best_model.pth --num_scenes 50
    python test_model_strength.py --model_path best_model.pth --visualize_scene data/raw/0_545
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Ensure project root is importable and use the current src/ layout
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GQEstimator

class SceneDataset:
    """
    Lightweight scene-level dataset compatible with this script.
    Loads each scene.npz under data/processed/<scene_id>/ and returns
    tensors for 'sdf', 'grasps', and 'scores'.
    """
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data_files = [
            p / 'scene.npz' for p in self.data_path.iterdir()
            if p.is_dir() and (p / 'scene.npz').exists()
        ]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx: int):
        scene_file = self.data_files[idx]
        with np.load(scene_file) as scene_data:
            sdf = torch.from_numpy(scene_data["sdf"]).float()
            grasps = torch.from_numpy(scene_data["grasps"]).float()
            scores = torch.from_numpy(scene_data["scores"]).float()
        return {"sdf": sdf, "grasps": grasps, "scores": scores}


class ComprehensiveModelTester:
    def __init__(self, model_path, device='cuda'):
        """Initialize the model tester with a pre-trained model."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model with same architecture as training
        self.model = GQEstimator(input_size=48, base_channels=16, fc_dims=[256, 128, 64])
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")

    def predict_scene_grasps(self, scene_data, batch_size=64):
        """Predict scores for all grasps in a scene."""
        sdf = scene_data['sdf'].to(self.device)
        grasps = scene_data['grasps'].to(self.device)
        true_scores = scene_data['scores']
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(grasps), batch_size):
                batch_grasps = grasps[i:i+batch_size]
                batch_size_actual = len(batch_grasps)
                
                # Expand SDF for batch
                sdf_batch = sdf.unsqueeze(0).expand(batch_size_actual, -1, -1, -1)
                
                # Predict
                batch_pred = self.model.forward_with_sdf(sdf_batch, batch_grasps)
                predictions.append(batch_pred.cpu())
        
        predicted_scores = torch.cat(predictions)
        return predicted_scores, true_scores

    def calculate_topk_rates(self, predicted_scores, true_scores, k_values=[1, 5, 10, 50]):
        """Calculate Top-K accuracy rates."""
        # Get ranking indices
        _, pred_ranking = torch.sort(predicted_scores, descending=True)
        _, true_ranking = torch.sort(true_scores, descending=True)
        
        results = {}
        
        for k in k_values:
            if k > len(predicted_scores):
                continue
                
            # Get top-k indices
            pred_topk = set(pred_ranking[:k].tolist())
            true_topk = set(true_ranking[:k].tolist())
            
            # Calculate overlap
            overlap = len(pred_topk.intersection(true_topk))
            topk_rate = overlap / k
            
            results[f'top{k}_rate'] = topk_rate
        
        return results

    def test_score_ordering(self, predicted_scores, true_scores):
        """Test if the model maintains the correct ordering of scores."""
        pred_np = predicted_scores.numpy()
        true_np = true_scores.numpy()
        
        # Calculate Spearman correlation (measures rank correlation)
        spearman_corr, spearman_p = spearmanr(pred_np, true_np)
        
        # Calculate Kendall's tau (another rank correlation measure)
        kendall_tau, kendall_p = kendalltau(pred_np, true_np)
        
        return {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p_value': kendall_p,
            'order_maintained': spearman_corr > 0.5  # Arbitrary threshold
        }

    def test_dataset(self, dataset, num_scenes=None):
        """Test model on multiple scenes."""
        if num_scenes is None:
            num_scenes = len(dataset)
        else:
            num_scenes = min(num_scenes, len(dataset))
        
        print(f"Testing model on {num_scenes} scenes...")
        
        all_topk_results = []
        all_ordering_results = []
        scene_results = []
        
        for scene_idx in tqdm(range(num_scenes), desc="Testing scenes"):
            scene_data = dataset[scene_idx]
            
            # Get predictions
            pred_scores, true_scores = self.predict_scene_grasps(scene_data)
            
            # Calculate metrics
            topk_results = self.calculate_topk_rates(pred_scores, true_scores)
            ordering_results = self.test_score_ordering(pred_scores, true_scores)
            
            all_topk_results.append(topk_results)
            all_ordering_results.append(ordering_results)
            
            # Store scene result
            scene_result = {
                'scene_idx': scene_idx,
                'num_grasps': len(pred_scores),
                'pred_scores': pred_scores.numpy(),
                'true_scores': true_scores.numpy(),
                'topk_accuracies': topk_results,
                'correlations': ordering_results
            }
            scene_results.append(scene_result)
        
        return all_topk_results, all_ordering_results, scene_results

    def print_results(self, all_topk_results, all_ordering_results, num_scenes):
        """Print comprehensive test results."""
        print("\n" + "="*60)
        print("MODEL STRENGTH TEST RESULTS")
        print("="*60)
        print(f"Tested on {num_scenes} scenes")
        print()
        
        # Top-K Accuracy Rates
        print("TOP-K ACCURACY RATES:")
        print("-"*30)
        
        topk_keys = sorted(all_topk_results[0].keys())
        for key in topk_keys:
            values = [result[key] for result in all_topk_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            k_num = key.replace('top', '').replace('_rate', '')
            print(f"Top-{k_num:2s}: {mean_val:.3f} ± {std_val:.3f}")
        
        print()
        
        # Score Ordering Results
        print("SCORE ORDERING ANALYSIS:")
        print("-"*30)
        
        spearman_values = [r['spearman_correlation'] for r in all_ordering_results 
                          if not np.isnan(r['spearman_correlation'])]
        kendall_values = [r['kendall_tau'] for r in all_ordering_results 
                         if not np.isnan(r['kendall_tau'])]
        
        print(f"Spearman Correlation: {np.mean(spearman_values):.3f} ± {np.std(spearman_values):.3f}")
        print(f"Kendall Tau:          {np.mean(kendall_values):.3f} ± {np.std(kendall_values):.3f}")
        
        # Count scenes where order is well maintained
        well_ordered = sum(1 for r in all_ordering_results if r['order_maintained'])
        print(f"Scenes with good ordering (ρ > 0.5): {well_ordered}/{num_scenes} ({well_ordered/num_scenes*100:.1f}%)")
        
        print()
        print("="*60)
        
        # Summary interpretation
        avg_spearman = np.mean(spearman_values)
        avg_top1 = np.mean([result['top1_rate'] for result in all_topk_results])
        avg_top10 = np.mean([result['top10_rate'] for result in all_topk_results])
        
        print("INTERPRETATION:")
        print("-"*15)
        
        if avg_spearman > 0.7:
            print("✅ EXCELLENT: Model maintains score ordering very well")
        elif avg_spearman > 0.5:
            print("✅ GOOD: Model maintains score ordering reasonably well") 
        elif avg_spearman > 0.3:
            print("⚠️  MODERATE: Model has some correlation with true ordering")
        else:
            print("❌ POOR: Model struggles to maintain correct score ordering")
        
        if avg_top1 > 0.8:
            print("✅ EXCELLENT: Very high Top-1 accuracy")
        elif avg_top1 > 0.6:
            print("✅ GOOD: Good Top-1 accuracy")
        elif avg_top1 > 0.4:
            print("⚠️  MODERATE: Moderate Top-1 accuracy")
        else:
            print("❌ POOR: Low Top-1 accuracy")
        
        if avg_top10 > 0.9:
            print("✅ EXCELLENT: Very high Top-10 accuracy")
        elif avg_top10 > 0.7:
            print("✅ GOOD: Good Top-10 accuracy")
        elif avg_top10 > 0.5:
            print("⚠️  MODERATE: Moderate Top-10 accuracy")
        else:
            print("❌ POOR: Low Top-10 accuracy")

    def create_visualizations(self, scene_results, save_path='test_results.png'):
        """Create visualization plots of the test results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Prediction Test Results', fontsize=16)
        
        # Extract data for plotting
        all_topk_results = [sr['topk_accuracies'] for sr in scene_results]
        all_correlations = [sr['correlations'] for sr in scene_results]
        
        # Top-K Accuracy Plot
        ax1 = axes[0, 0]
        k_values = []
        accuracies = []
        errors = []
        
        topk_keys = sorted(all_topk_results[0].keys())
        for key in topk_keys:
            k_val = int(key.replace('top', '').replace('_rate', ''))
            k_values.append(k_val)
            values = [result[key] for result in all_topk_results]
            accuracies.append(np.mean(values))
            errors.append(np.std(values))
        
        ax1.errorbar(k_values, accuracies, yerr=errors, marker='o', capsize=5)
        ax1.set_xlabel('K Value')
        ax1.set_ylabel('Top-K Accuracy')
        ax1.set_title('Top-K Accuracy vs K')
        ax1.grid(True)
        ax1.set_ylim(0, 1)
        
        # Correlation Distribution
        ax2 = axes[0, 1]
        spearman_values = [sr['correlations']['spearman_correlation'] 
                          for sr in scene_results 
                          if not np.isnan(sr['correlations']['spearman_correlation'])]
        
        ax2.hist(spearman_values, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_ylabel('Number of Scenes')
        ax2.set_title('Distribution of Spearman Correlations')
        ax2.axvline(np.mean(spearman_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(spearman_values):.3f}')
        ax2.legend()
        
        # Example scene prediction vs truth
        ax3 = axes[1, 0]
        example_scene = scene_results[0]  # First scene as example
        pred_scores = example_scene['pred_scores']
        true_scores = example_scene['true_scores']
        
        ax3.scatter(true_scores, pred_scores, alpha=0.6)
        ax3.plot([true_scores.min(), true_scores.max()], 
                [true_scores.min(), true_scores.max()], 'r--', lw=2)
        ax3.set_xlabel('True Scores')
        ax3.set_ylabel('Predicted Scores')
        ax3.set_title(f'Predictions vs Truth (Scene {example_scene["scene_idx"]})')
        
        # Top-K accuracy comparison
        ax4 = axes[1, 1]
        k_names = [f'Top-{k}' for k in k_values]
        
        bars = ax4.bar(k_names, accuracies, alpha=0.7)
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Top-K Accuracy Comparison')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to '{save_path}'")

    def visualize_top10_grasps(self, scene_path):
        """Visualize top 10 grasps using PyBullet."""
        try:
            import pybullet
        except ImportError:
            print("❌ PyBullet not available. Install with: pip install pybullet")
            return
        
        scene_path = Path(scene_path)
        mesh_path = scene_path / "mesh.obj"
        recording_path = scene_path / "recording.npz"
        
        # Check if files exist
        if not mesh_path.exists():
            print(f"❌ mesh.obj not found at {mesh_path}")
            return
        
        if not recording_path.exists():
            print(f"❌ recording.npz not found at {recording_path}")
            return
        
        print(f"🎯 Visualizing top 10 grasps from {scene_path}")
        
        # Connect to PyBullet
        pybullet.connect(pybullet.GUI)
        
        # Set up camera for better visualization
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Load hand
        print("Loading hand model...")
        hand_id = pybullet.loadURDF(
            "urdfs/dlr2.urdf",
            globalScaling=1,
            basePosition=[0, 0, 0],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
        )
        
        # Load object
        print("Loading object mesh...")
        visualShapeId = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=str(mesh_path),
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0, 0, 0],
            meshScale=1
        )
        
        object_id = pybullet.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeId,
            baseCollisionShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # Load grasps
        print("Loading grasp data...")
        data = np.load(recording_path)
        grasps = data["grasps"]
        scores = data["scores"]
        
        print(f"Total grasps in dataset: {grasps.shape[0]}")
        print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
        
        # Sort by grasp score (descending) and get top 10
        sorted_indices = np.argsort(scores)[::-1]
        top10_indices = sorted_indices[:10]
        
        print(f"\nVisualizing top 10 grasps:")
        print("-" * 50)
        
        # Visualize top grasps
        for rank, grasp_idx in enumerate(top10_indices):
            grasp = grasps[grasp_idx]
            score = scores[grasp_idx]
            
            print(f"Rank {rank + 1:2d}: Grasp {grasp_idx:3d} - Score: {score:8.4f}")
            
            # Set hand pose (position and orientation)
            hand_position = grasp[:3]
            hand_orientation = grasp[3:7]  # quaternion [x, y, z, w]
            
            pybullet.resetBasePositionAndOrientation(
                bodyUniqueId=hand_id,
                posObj=hand_position,
                ornObj=hand_orientation
            )
            
            # Set joint angles for fingers
            joint_indices = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]
            finger_angles = grasp[7:19]  # 12 finger joint values
            
            for k, joint_idx in enumerate(joint_indices):
                pybullet.resetJointState(
                    hand_id,
                    jointIndex=joint_idx,
                    targetValue=finger_angles[k],
                    targetVelocity=0
                )
                
                # Set coupled joint for certain fingers
                if joint_idx in [3, 9, 15, 21]:
                    pybullet.resetJointState(
                        hand_id,
                        jointIndex=joint_idx + 1,
                        targetValue=finger_angles[k],
                        targetVelocity=0
                    )
            
            # Clear previous debug text and add new text
            pybullet.removeAllUserDebugItems()
            text_position = [hand_position[0], hand_position[1], hand_position[2] + 0.2]
            pybullet.addUserDebugText(
                text=f"Rank {rank + 1} | Score: {score:.4f}",
                textPosition=text_position,
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                lifeTime=0
            )
            
            print(f"  Position: [{hand_position[0]:.3f}, {hand_position[1]:.3f}, {hand_position[2]:.3f}]")
            print(f"  Orientation: [{hand_orientation[0]:.3f}, {hand_orientation[1]:.3f}, {hand_orientation[2]:.3f}, {hand_orientation[3]:.3f}]")
            print("  Press Enter to continue to next grasp, or 'q' + Enter to quit...")
            
            user_input = input()
            if user_input.lower() == 'q':
                break
        
        print("\nVisualization completed!")
        
        # Summary statistics
        top_scores = scores[top10_indices]
        print(f"\nTop 10 grasp statistics:")
        print(f"  Highest score: {top_scores.max():.4f}")
        print(f"  Lowest score:  {top_scores.min():.4f}")
        print(f"  Mean score:    {top_scores.mean():.4f}")
        print(f"  Std score:     {top_scores.std():.4f}")
        
        # Disconnect from PyBullet
        pybullet.disconnect()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model prediction testing and visualization')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--num_scenes', type=int, default=50,
                       help='Number of scenes to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for testing')
    parser.add_argument('--visualize_plots', action='store_true',
                       help='Create and show result plots')
    parser.add_argument('--visualize_scene', type=str, default=None,
                       help='Path to scene directory for top 10 grasp visualization')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to pickle file')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"❌ Model file '{args.model_path}' not found!")
        return
    
    # Initialize tester
    print("Initializing model tester...")
    tester = ComprehensiveModelTester(args.model_path, device=args.device)
    
    # Run dataset testing
    if Path(args.data_path).exists():
        print("Loading dataset...")
        dataset = SceneDataset(Path(args.data_path))
        print(f"Found {len(dataset)} scenes in dataset")
        
        num_scenes = min(args.num_scenes, len(dataset))
        
        # Run tests
        all_topk_results, all_ordering_results, scene_results = tester.test_dataset(dataset, num_scenes)
        
        # Print results
        tester.print_results(all_topk_results, all_ordering_results, num_scenes)
        
        # Create visualizations if requested
        if args.visualize_plots:
            tester.create_visualizations(scene_results)
        
        # Save results if requested
        if args.save_results:
            with open('comprehensive_test_results.pkl', 'wb') as f:
                pickle.dump({
                    'topk_results': all_topk_results,
                    'ordering_results': all_ordering_results,
                    'scene_results': scene_results
                }, f)
            print("✅ Results saved to 'comprehensive_test_results.pkl'")
    
    else:
        print(f"⚠️  Data directory '{args.data_path}' not found, skipping dataset tests")
    
    # Run grasp visualization if requested
    if args.visualize_scene:
        tester.visualize_top10_grasps(args.visualize_scene)
    
    print("\n🎉 Testing completed successfully!")
    print("\nUsage examples:")
    print("# Basic testing:")
    print(f"python test_model_strength.py --num_scenes 100")
    print("# With plots:")
    print(f"python test_model_strength.py --num_scenes 100 --visualize_plots")
    print("# With grasp visualization:")
    print(f"python test_model_strength.py --visualize_scene data/raw/0_545")
    print("# Full test with everything:")
    print(f"python test_model_strength.py --num_scenes 100 --visualize_plots --visualize_scene data/raw/0_545 --save_results")


if __name__ == "__main__":
    main() 