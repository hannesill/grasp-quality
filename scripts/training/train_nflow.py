import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import GraspDataset
from src.model_nflow import create_nflow_model

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Normalizing Flow on Grasp Configurations")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use. Defaults to all.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation data split ratio.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--model_path', type=str, default='nflow_model.pth', help='Path to save the trained model')
    parser.add_argument('--wandb_entity', type=str, default='tairo', help='WandB entity')
    parser.add_argument('--project_name', type=str, default='adlr-nflow', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--input_dim', type=int, default=19, help='Dimension of the grasp configuration')
    parser.add_argument('--num_flow_layers', type=int, default=5, help='Number of flow layers')
    parser.add_argument('--hidden_features', type=int, default=64, help='Hidden features in the flow transforms')
    return parser.parse_args()

def main(args):
    wandb.init(
        entity=args.wandb_entity,
        project=args.project_name,
        name=args.run_name,
        config=args
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = Path(args.data_path)
    dataset = GraspDataset(data_path)

    num_available = len(dataset)
    print(f"Total samples available: {num_available}")

    num_to_use = num_available if args.num_samples is None else min(args.num_samples, num_available)
    print(f"Using {num_to_use} samples.")

    indices = list(range(num_available))
    random.shuffle(indices)
    indices = indices[:num_to_use]

    val_size = int(num_to_use * args.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(f"Train dataset size: {len(train_set)}, Validation dataset size: {len(val_set)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=False
    )

    model = create_nflow_model(
        input_dim=args.input_dim,
        num_layers=args.num_flow_layers,
        hidden_features=args.hidden_features
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb.watch(model, log="all", log_freq=100)

    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        model.train()
        total_train_loss = 0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for batch in pbar:
            optimizer.zero_grad()
            
            grasp_batch = batch['grasp'].to(device)
            
            loss = -model.log_prob(inputs=grasp_batch).mean()
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_steps += 1
            
            running_loss = total_train_loss / num_steps
            pbar.set_postfix(loss=f'{running_loss:.4f}')

        avg_train_loss = total_train_loss / num_steps

        model.eval()
        total_val_loss = 0
        num_steps = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for batch in pbar_val:
                grasp_batch = batch['grasp'].to(device)
                loss = -model.log_prob(inputs=grasp_batch).mean()
                total_val_loss += loss.item()
                num_steps += 1
                
                running_val_loss = total_val_loss / num_steps
                pbar_val.set_postfix(val_loss=f'{running_val_loss:.4f}')

        avg_val_loss = total_val_loss / num_steps
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{args.epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_nflow_model.pth')
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
        })
        
    model_path = f"{wandb.run.dir}/{args.model_path}"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model saved to best_nflow_model.pth")
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)
