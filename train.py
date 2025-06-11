import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from dataset import GraspDataset, GraspBatchIterableDataset
from model import GQEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="Train Grasp Quality Estimator")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_size', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for the CNN')
    parser.add_argument('--fc_dims', nargs='+', type=int, default=[256, 128, 64], help='Dimensions of FC layers in the head')
    parser.add_argument('--grasp_batch_size', type=int, default=32, help='Batch size for grasps within a scene')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--project_name', type=str, default='grasp-quality', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    return parser.parse_args()

def main(args):
    # --- WandB Initialization ---
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=args
    )

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    data_path = Path(args.data_path)
    dataset = GraspDataset(data_path)

    num_samples = len(dataset)
    train_size = int(num_samples * (1 - args.val_split))
    val_size = num_samples - train_size
    print(f"Total scenes: {num_samples}, Train scenes: {train_size}, Val scenes: {val_size}")

    indices = list(range(num_samples))
    random.seed(42)
    random.shuffle(indices)

    train_indices = indices[:args.train_size]
    val_indices = indices[-args.val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(f"Train dataset size: {len(train_set)}, Validation dataset size: {len(val_set)}")

    train_iterable_dataset = GraspBatchIterableDataset(train_set, grasp_batch_size=args.grasp_batch_size, shuffle_scenes=True)
    val_iterable_dataset = GraspBatchIterableDataset(val_set, grasp_batch_size=args.grasp_batch_size, shuffle_scenes=False)
    
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_iterable_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(val_iterable_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=True if args.num_workers > 0 else False)

    # --- Model, Optimizer, Loss ---
    model = GQEstimator(input_size=48, base_channels=args.base_channels, fc_dims=args.fc_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    wandb.watch(model, criterion, log="all", log_freq=100)

    # --- Training Loop ---
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        total_train_grasps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for sdf, grasp_batch, score_batch in pbar:
            optimizer.zero_grad()
            
            sdf, grasp_batch, score_batch = sdf.to(device), grasp_batch.to(device), score_batch.to(device)

            num_grasps_in_batch = grasp_batch.shape[0]
            
            # 1. Encode SDF
            sdf_features = model.encode_sdf(sdf)

            # 2. Expand features for the grasp batch
            expanded_sdf_features = sdf_features.expand(num_grasps_in_batch, -1)

            # 3. Concatenate features
            flattened_features = torch.cat([expanded_sdf_features, grasp_batch], dim=1)

            # 4. Predict grasp quality and compute loss
            pred_quality = model(flattened_features)
            loss = criterion(pred_quality, score_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * args.grasp_batch_size
            total_train_grasps += args.grasp_batch_size
            
            if total_train_grasps > 0:
                running_loss = total_train_loss / total_train_grasps
                pbar.set_postfix(loss=f'{running_loss:.4f}')
        
        avg_train_loss = total_train_loss / total_train_grasps if total_train_grasps > 0 else 0

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        total_val_grasps = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for sdf, grasp_batch, score_batch in pbar_val:
                sdf, grasp_batch, score_batch = sdf.to(device), grasp_batch.to(device), score_batch.to(device)

                num_grasps_in_batch = grasp_batch.shape[0]
                
                # 1. Encode SDF
                sdf_features = model.encode_sdf(sdf)

                # 2. Expand features for the grasp batch
                expanded_sdf_features = sdf_features.expand(num_grasps_in_batch, -1)

                # 3. Concatenate features
                flattened_features = torch.cat([expanded_sdf_features, grasp_batch], dim=1)

                # 4. Predict grasp quality and compute loss
                pred_quality = model(flattened_features)
                loss = criterion(pred_quality, score_batch)

                total_val_loss += loss.item() * args.grasp_batch_size
                total_val_grasps += args.grasp_batch_size
                
                if total_val_grasps > 0:
                    running_val_loss = total_val_loss / total_val_grasps if total_val_grasps > 0 else 0
                    pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / total_val_grasps if total_val_grasps > 0 else 0
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

    # --- Save Model ---
    model_path = f"{wandb.run.dir}/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)