#!/usr/bin/env python3
"""
Test the strength of model predictions by evaluating ranking accuracy.

This script tests if the model can maintain the correct order of grasp scores
and calculates Top1, Top5, Top10, and Top50 accuracy rates.

Usage:
    python test_model_strength.py --model_path best_model.pth --num_scenes 50
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

from model import GQEstimator
from dataset import SceneDataset


def load_model(model_path, device='cuda'):
    """Load the trained model."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model with same architecture as training
    model = GQEstimator(input_size=48, base_channels=16, fc_dims=[256, 128, 64])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


def predict_scene_grasps(model, scene_data, device, batch_size=64):
    """Predict scores for all grasps in a scene."""
    sdf = scene_data['sdf'].to(device)
    grasps = scene_data['grasps'].to(device)
    true_scores = scene_data['scores']
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(grasps), batch_size):
            batch_grasps = grasps[i:i+batch_size]
            batch_size_actual = len(batch_grasps)
            
            # Expand SDF for batch
            sdf_batch = sdf.unsqueeze(0).expand(batch_size_actual, -1, -1, -1)
            
            # Predict
            batch_pred = model.forward_with_sdf(sdf_batch, batch_grasps)
            predictions.append(batch_pred.cpu())
    
    predicted_scores = torch.cat(predictions)
    return predicted_scores, true_scores


def calculate_topk_rates(predicted_scores, true_scores, k_values=[1, 5, 10, 50]):
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


def test_score_ordering(predicted_scores, true_scores):
    """Test if the model maintains the correct ordering of scores."""
    # Calculate Spearman correlation (measures rank correlation)
    spearman_corr, p_value = spearmanr(predicted_scores.numpy(), true_scores.numpy())
    
    # Calculate Kendall's tau (another rank correlation measure)
    from scipy.stats import kendalltau
    kendall_tau, kendall_p = kendalltau(predicted_scores.numpy(), true_scores.numpy())
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_p_value': p_value,
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p,
        'order_maintained': spearman_corr > 0.5  # Arbitrary threshold
    }


def main():
    parser = argparse.ArgumentParser(description='Test model prediction strength')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data')
    parser.add_argument('--num_scenes', type=int, default=50,
                       help='Number of scenes to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load model and dataset
    print("Loading model...")
    model, device = load_model(args.model_path, args.device)
    
    print("Loading dataset...")
    dataset = SceneDataset(Path(args.data_path))
    num_scenes = min(args.num_scenes, len(dataset))
    
    print(f"Testing on {num_scenes} scenes...")
    
    # Test results
    all_topk_results = []
    all_ordering_results = []
    
    print("\nTesting scenes...")
    for scene_idx in tqdm(range(num_scenes)):
        scene_data = dataset[scene_idx]
        
        # Get predictions
        pred_scores, true_scores = predict_scene_grasps(model, scene_data, device)
        
        # Calculate metrics
        topk_results = calculate_topk_rates(pred_scores, true_scores)
        ordering_results = test_score_ordering(pred_scores, true_scores)
        
        all_topk_results.append(topk_results)
        all_ordering_results.append(ordering_results)
    
    # Aggregate results
    print("\n" + "="*60)
    print("MODEL STRENGTH TEST RESULTS")
    print("="*60)
    print(f"Tested on {num_scenes} scenes")
    print()
    
    # Top-K Accuracy Rates
    print("TOP-K ACCURACY RATES:")
    print("-"*30)
    
    topk_keys = all_topk_results[0].keys()
    for key in sorted(topk_keys):
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


if __name__ == "__main__":
    main()
