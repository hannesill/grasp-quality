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
from src.dataset_sdf import SDFDataset
from src.model import ObjectEncoder
from src.model_diff import DiffusionModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Model for Grasp Generation")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use. Defaults to all.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation data split ratio.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--encoder_path', type=str, default="checkpoints/object_encoder.pth", help='Path to pretrained encoder checkpoint')
    parser.add_argument('--wandb_entity', type=str, default='tairo', help='WandB entity')
    parser.add_argument('--project_name', type=str, default='adlr-diffusion', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for the object encoder')
    parser.add_argument('--sdf_feature_dim', type=int, default=3456, help='Dimension of the SDF features')

    return parser.parse_args()

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

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
    dataset = GraspDataset(data_path, preload=True)

    num_available = len(dataset)
    num_to_use = num_available if args.num_samples is None else min(args.num_samples, num_available)
    
    indices = list(range(num_available))
    random.shuffle(indices)
    indices = indices[:num_to_use]

    val_size = int(num_to_use * args.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=pin_memory, persistent_workers=True)
    
    object_encoder = ObjectEncoder(base_channels=args.base_channels).to(device)
    if args.encoder_path:
        object_encoder.load_state_dict(torch.load(args.encoder_path, weights_only=True, map_location=device))
    object_encoder.eval()
    for param in object_encoder.parameters():
        param.requires_grad = False
    object_encoder.to(device)

    model = DiffusionModel().to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    betas = linear_beta_schedule(args.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    # Pre-compute SDF features
    print("Pre-computing SDF features...")
    sdf_features_cache = torch.zeros(len(dataset), args.sdf_feature_dim, device=device)
    with torch.no_grad():
        for i, sdf in enumerate(tqdm(dataset.sdfs, desc="Encoding SDFs")):
            sdf = torch.from_numpy(sdf).float().to(device)
            features = object_encoder(sdf.unsqueeze(0)).view(-1)
            sdf_features_cache[i] = features
    
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for batch in pbar:
            optimizer.zero_grad()
            
            grasp_batch = batch['grasp'].to(device)
            scene_idx_batch = batch['scene_idx']

            with torch.no_grad():
                sdf_features = sdf_features_cache[scene_idx_batch].to(device)

            t = torch.randint(0, args.timesteps, (grasp_batch.size(0),), device=device).long()
            
            noise = torch.randn_like(grasp_batch)
            alpha_t = alphas_cumprod[t].view(-1, 1)
            
            noisy_grasp = alpha_t.sqrt() * grasp_batch + (1 - alpha_t).sqrt() * noise
            
            predicted_noise = model(noisy_grasp, t, sdf_features)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f'{total_train_loss / (pbar.n + 1):.4f}')

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation")
            for batch in pbar_val:
                grasp_batch = batch['grasp'].to(device)
                scene_idx_batch = batch['scene_idx']

                sdf_features = sdf_features_cache[scene_idx_batch].to(device)

                t = torch.randint(0, args.timesteps, (grasp_batch.size(0),), device=device).long()
                
                noise = torch.randn_like(grasp_batch)
                alpha_t = alphas_cumprod[t].view(-1, 1)

                noisy_grasp = alpha_t.sqrt() * grasp_batch + (1 - alpha_t).sqrt() * noise
                
                predicted_noise = model(noisy_grasp, t, sdf_features)
                loss = criterion(predicted_noise, noise)
                
                total_val_loss += loss.item()
                pbar_val.set_postfix(val_loss=f'{total_val_loss / (pbar_val.n + 1):.4f}')

        avg_val_loss = total_val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{wandb.run.dir}/best_model.pth")

    model_path = f"{wandb.run.dir}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)
