import argparse
import json
import random
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_splits(data_path, train_ratio=0.8, val_ratio=0.1, output_path='data/splits.json'):
    """
    Scans a directory for scene data, splits the scenes into train, validation, and test sets,
    and saves the split information to a JSON file.

    Args:
        data_path (str): The path to the processed data directory containing scene subdirectories.
        train_ratio (float): The proportion of data to allocate to the training set.
        val_ratio (float): The proportion of data to allocate to the validation set.
        output_path (str): The file path where the JSON split file will be saved.
    """
    data_path = Path(data_path)
    scene_dirs = [d for d in data_path.iterdir() if d.is_dir() and (d / 'scene.npz').exists()]
    scene_ids = sorted([d.name for d in scene_dirs])
    
    # Ensure reproducibility
    random.seed(42)
    random.shuffle(scene_ids)
    
    num_scenes = len(scene_ids)
    train_end = int(num_scenes * train_ratio)
    val_end = train_end + int(num_scenes * val_ratio)
    
    train_ids = scene_ids[:train_end]
    val_ids = scene_ids[train_end:val_end]
    test_ids = scene_ids[val_end:]
    
    splits = {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids)
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Splits created and saved to {output_path}")
    print(f"Train scenes: {len(train_ids)}")
    print(f"Validation scenes: {len(val_ids)}")
    print(f"Test scenes: {len(test_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits for the dataset.")
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio.')
    parser.add_argument('--output_path', type=str, default='data/splits.json', help='Path to save the splits file.')
    args = parser.parse_args()
    
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("The sum of train_ratio and val_ratio cannot be greater than 1.")

    create_splits(args.data_path, args.train_ratio, args.val_ratio, args.output_path) 